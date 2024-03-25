import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
# import tensorflow_probability as tfp
import tensorflow as tf
from dipy.align.reslice import reslice

import wandb

class WandbImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, val_dataset, layer_idx=64, num_images=5, log_freq=10):
        # Initialize with the VQVAE model and validation dataset
        self.model = model
        self.val_dataset = val_dataset
        self.layer_idx = layer_idx  # The specific layer to extract and log
        self.num_images = num_images
        self.log_freq = log_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_freq == 0:
            # Generate a batch from the validation dataset
            for x, mask, _ in self.val_dataset.take(1):
                break
            x = tf.concat([x, mask], axis=-1)
            # Ensure we only use a subset for visualization to save resources
            x_subset = x[:self.num_images]

            # Get the reconstructions from the VQVAE model
            img_reconstructed, _ = self.model(x_subset, training=False)
            reconstructions, mask_reconstructed = tf.split(img_reconstructed, num_or_size_splits=2, axis=-1)
            # Prepare images for logging
            images = []
            for i in range(self.num_images):
                # Extract the specific layer from the original and reconstructed images
                original_slice = x[i,:,:,self.layer_idx,0].numpy()
                reconstructed_slice = reconstructions[i,:,:,self.layer_idx,0].numpy()

                # Normalize and convert to uint8
                original_slice = np.clip(original_slice * 255.0, 0, 255).astype(np.uint8)
                reconstructed_slice = np.clip(reconstructed_slice * 255.0, 0, 255).astype(np.uint8)

                # Plot original and reconstructed slices side by side
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(original_slice, cmap='gray')
                axes[0].set_title('Original')
                axes[0].axis('off')
                axes[1].imshow(reconstructed_slice, cmap='gray')
                axes[1].set_title('Reconstructed')
                axes[1].axis('off')
                plt.tight_layout()

                # Convert the plot to an image array
                fig.canvas.draw()
                img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                # Close the plot to free memory
                plt.close(fig)

                # Append the side-by-side image to the list for logging
                images.append(wandb.Image(img_array, caption=f"Epoch {epoch+1} Original & Reconstructed"))

            # Log the images to W&B
            wandb.log({"Original & Reconstructions": images}, commit=False)


class ICNR:
    def __init__(self, initializer, scale=2):
        """ICNR initializer for checkerboard artifact free transpose convolution

        Code adapted from https://github.com/kostyaev/ICNR
        Discussed at https://github.com/Lasagne/Lasagne/issues/862
        Original paper: https://arxiv.org/pdf/1707.02937.pdf

        Parameters
        ----------
        initializer : Initializer
            Initializer used for kernels (glorot uniform, etc.)
        scale : iterable of two integers, or a single integer
            Stride of the transpose convolution
            (a.k.a. scale factor of sub pixel convolution)
        """
        self.scale = 2  # normalize_tuple(scale, 2, "scale")
        self.initializer = initializer

    def __call__(self, shape, dtype):
        if self.scale == 1:
            return self.initializer(shape)
        h, w, d, o, i = shape
        new_shape = (h//self.scale, w//self.scale, d//self.scale, o, i)
        x = self.initializer(new_shape, dtype).numpy()
        x = x.reshape(new_shape[:3]+(o*i,))
        x, _ = reslice(x, np.eye(4), (1, 1, 1), (0.5, 0.5, 0.5))
        x = x.reshape(shape[:3]+(o, i))
        return x


class ReplaceCodebookCallback(tf.keras.callbacks.Callback):
    def __init__(self, vq_layer, batch_size, frequency=10):
        self.vq_layer = vq_layer
        self.frequency = frequency
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.frequency == 0:
            print(f"Epoch {epoch+1}: Replacing unused codebooks.")
            self.vq_layer.replace_unused_codebooks(self.batch_size)


class VectorQuantizer(keras.Model):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta
        self.eps = 1e-10
        self.discarding_threshold = 0.05
        # Initialize the embeddings which we will quantize.
        w_init = tf.keras.initializers.HeUniform()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )
        self.codebooks_used = tf.Variable(tf.zeros([num_embeddings], dtype=tf.int32), trainable=False, name='codebooks_used')

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)

        avg_probs = tf.reduce_mean(encodings, axis=0)
        perplexity = tf.exp(-tf.reduce_sum(avg_probs * tf.math.log(avg_probs + self.eps)))

        self.codebooks_used.assign_add(tf.cast(tf.math.bincount(tf.cast(encoding_indices, tf.int32), minlength=self.num_embeddings), tf.int32))

        return quantized, perplexity

    def get_code_indices(self, flattened_inputs, distribution=False):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )
        if distribution:
            return distances
        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices
    
    def replace_unused_codebooks(self, num_batches):
        usage_rates = self.codebooks_used / num_batches

        unused_indices = tf.where(usage_rates < self.discarding_threshold)
        used_indices = tf.where(usage_rates >= self.discarding_threshold)

        unused_count = tf.shape(unused_indices)[0]
        used_count = tf.shape(used_indices)[0]
        print("Used: ", used_count.numpy(), "Unused:", unused_count.numpy())

        if used_count == 0:
            print(f'####### used_indices equals zero / shuffling whole codebooks ######')
            # If no codebooks are used, add noise to all embeddings
            self.embeddings.assign_add(self.eps * tf.random.normal(self.embeddings.shape))
        else:
            # Gather used codebooks
            used_codebooks = tf.gather(self.embeddings, used_indices[:, 0], axis=0)

            # Ensure the shape of used_codebooks matches the number of unused codebooks by repeating or truncating
            if used_count < unused_count:
                repeat_factor = (unused_count // used_count) + 1
                used_codebooks = tf.tile(used_codebooks, [repeat_factor, 1])[:unused_count, :]
                used_codebooks = tf.random.shuffle(used_codebooks)
            
            # Generate noise with the same shape as used_codebooks
            noise = self.eps * tf.random.normal(shape=tf.shape(used_codebooks))

            # Add noise to used_codebooks
            updated_values = used_codebooks + noise

            # Update the embeddings tensor with the new values for unused codebooks
            unused_indices = tf.cast(unused_indices, dtype=tf.int32)
            self.embeddings = tf.tensor_scatter_nd_update(self.embeddings, unused_indices, updated_values)

        print(f'************* Replaced ' + str(unused_count.numpy()) + f' codebooks *************')
        # Reset the codebooks usage tracker
        self.codebooks_used.assign(tf.zeros_like(self.codebooks_used))


class VQVAEResidualUnit(keras.Model):
    def __init__(self, input_channels, num_res_channels, act='relu'):
        super().__init__()
        self.num_res_channels = num_res_channels
        self.input_channels = input_channels
        self.act = act

        self.conv1 = layers.Conv3D(
            self.num_res_channels, 3, activation=self.act, strides=1, padding='same')
        self.conv2 = tf.keras.Sequential()
        self.conv2.add(layers.Conv3D(self.input_channels,
                       3, strides=1, padding='same'))
        self.conv2.add(tf.keras.layers.BatchNormalization())
        self.conv2.add(tf.keras.layers.PReLU())

    def call(self, x):
        return tf.keras.layers.ReLU()(x+self.conv2(self.conv1(x)))


class Encoder(keras.Model):
    """
    Encoder module for VQ-VAE.

    Args:
        spatial_dims: number of spatial spatial_dims.
        in_channels: number of input channels.
        out_channels: number of channels in the latent space (embedding_dim).
        num_channels: number of channels at each level.
        num_res_layers: number of sequential residual layers at each level.
        num_res_channels: number of channels in the residual layers at each level.
        downsample_parameters: A Tuple of Tuples for defining the downsampling convolutions. Each Tuple should hold the
            following information stride (int), kernel_size (int), dilation (int) and padding (int).
        dropout: dropout ratio.
        act: activation type and arguments.
    """

    def __init__(self, in_channels, out_channels, num_channels, num_res_layers, num_res_channels, downsample_parameters, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_channels = num_channels
        self.num_res_layers = num_res_layers
        self.num_res_channels = num_res_channels
        self.downsample_parameters = downsample_parameters
        self.dropout = dropout

        self.blocks = tf.keras.Sequential()

        for i in range(len(self.num_channels)):
            if i == 0:
                self.blocks.add(
                    layers.Conv3D(self.num_channels[i], self.downsample_parameters[i][1], strides=self.downsample_parameters[i][0],
                                  dilation_rate=self.downsample_parameters[i][2],
                                  padding=self.downsample_parameters[i][3],
                                  input_shape=(128, 128, 128, self.in_channels)
                                  ))
            else:
                self.blocks.add(
                    layers.Conv3D(self.num_channels[i], self.downsample_parameters[i][1], strides=self.downsample_parameters[i][0],
                                  dilation_rate=self.downsample_parameters[i][2],
                                  padding=self.downsample_parameters[i][3],
                                  # input_shape=(128,128,128,1)
                                  ))

            if i > 0 and self.dropout:
                self.blocks.add(layers.Dropout(self.dropout))
            self.blocks.add(layers.ReLU())

            for _ in range(self.num_res_layers):
                self.blocks.add(
                    VQVAEResidualUnit(
                        input_channels=self.num_channels[i],
                        num_res_channels=self.num_res_channels[i],
                        # act=self.act,
                        # dropout=self.dropout,
                    )
                )

        self.blocks.add(
            layers.Conv3D(self.out_channels, 3, strides=1, padding='same')
        )
        if self.dropout:
            self.blocks.add(layers.Dropout(self.dropout))
        self.blocks.add(layers.PReLU())
        #self.blocks = nn.ModuleList(self.blocks)

    def call(self, x):
        y = self.blocks(x)
        return y


class Decoder(keras.Model):
    """
    Decoder module for VQ-VAE.

    Args:
        in_channels: number of channels in the latent space (embedding_dim).
        out_channels: number of output channels.
        num_channels: number of channels at each level.
        num_res_layers: number of sequential residual layers at each level.
        num_res_channels: number of channels in the residual layers at each level.
        upsample_parameters: A Tuple of Tuples for defining the upsampling convolutions. Each Tuple should hold the
            following information stride (int), kernel_size (int), dilation (int), padding (int), output_padding (int).
        dropout: dropout ratio.
        act: activation type and arguments.
        output_act: activation type and arguments for the output.
    """

    def __init__(
            self,
            in_channels, out_channels, num_channels, num_res_layers, num_res_channels, upsample_parameters,
            dropout, output_act, kernel_resize=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_channels = num_channels
        self.num_res_layers = num_res_layers
        self.num_res_channels = num_res_channels
        self.upsample_parameters = upsample_parameters
        self.dropout = dropout
        #self.act = act
        self.output_act = output_act
        self.ICNR = kernel_resize

        reversed_num_channels = list(reversed(self.num_channels))

        self.blocks = []

        self.blocks.append(
            layers.Conv3D(reversed_num_channels[0], 3, strides=1, padding='same'))
        if self.dropout:
            self.blocks.append(layers.Dropout(self.dropout))
        self.blocks.append(layers.PReLU())

        reversed_num_res_channels = list(reversed(self.num_res_channels))
        for i in range(len(self.num_channels)):
            for _ in range(self.num_res_layers):
                self.blocks.append(
                    VQVAEResidualUnit(
                        input_channels=reversed_num_channels[i],
                        num_res_channels=reversed_num_res_channels[i]
                        # act=self.act,
                        # dropout=self.dropout,
                    )
                )

            out = self.out_channels if i == len(
                self.num_channels) - 1 else reversed_num_channels[i + 1]

            if self.ICNR:
                kernel_initializer = ICNR(
                    tf.keras.initializers.GlorotNormal(), scale=self.upsample_parameters[i][0])
            else:
                kernel_initializer = 'glorot_uniform'
            self.blocks.append(
                layers.Conv3DTranspose(out, self.upsample_parameters[i][1], strides=self.upsample_parameters[i][0],
                                       dilation_rate=self.upsample_parameters[i][2],
                                       padding=self.upsample_parameters[i][3],
                                       # output_padding=self.upsample_parameters[i][4]
                                       kernel_initializer=kernel_initializer))
            if i != len(self.num_channels) - 1:
                if self.dropout:
                    self.blocks.append(layers.Dropout(self.dropout))
                self.blocks.append(layers.ReLU())

        if self.output_act:
            self.blocks.append(layers.ReLU())

        #self.self.blocks = nn.ModuleList(self.blocks)

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class VQVAE(keras.Model):
    def __init__(self, in_channels, out_channels, num_channels, num_res_layers, num_res_channels,
                 downsample_parameters=(
                     (2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1)),
                 upsample_parameters=(
                     (2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
                 num_embeddings=128,
                 embedding_dim=64,
                 dropout=0.1,
                 act='relu',
                 output_act=None,
                 num_gpus=2,
                 kernel_resize=False):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_channels = num_channels
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_res_layers = num_res_layers
        self.num_res_channels = num_res_channels
        self.num_gpus = num_gpus

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=embedding_dim,
            num_channels=num_channels,
            num_res_layers=num_res_layers,
            num_res_channels=num_res_channels,
            downsample_parameters=downsample_parameters,
            dropout=dropout,
            # act=act,
        )

        self.decoder = Decoder(
            in_channels=embedding_dim,
            out_channels=out_channels,
            num_channels=num_channels,
            num_res_layers=num_res_layers,
            num_res_channels=num_res_channels,
            upsample_parameters=upsample_parameters,
            dropout=dropout,
            # act=act,
            output_act=output_act,
            kernel_resize=kernel_resize
        )

        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconst_loss")
        self.quantize_loss_tracker = keras.metrics.Mean(name="quantize_loss")
        self.perplexity_tracker = keras.metrics.Mean(name="perplexity")
        self.ssim_tracker = keras.metrics.Mean(name="SSIM")
        self.psnr_tracker = keras.metrics.Mean(name="PSNR")

    def call(self, x):
        encoded_inputs = self.encoder(x)
        quantized, perplexity = self.quantizer(encoded_inputs)
        decoder_outputs = self.decoder(quantized)
        return decoder_outputs, perplexity

    def call_2(self, x):
        encoded_inputs = self.encoder(x)
        quantized, perplexity = self.quantizer(encoded_inputs)
        tf.print(quantized.shape)
        return quantized

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.reconstruction_loss_tracker,
            self.quantize_loss_tracker,
            self.perplexity_tracker,
            self.ssim_tracker,
            self.psnr_tracker
        ]

    def train_step(self, inputs):
        img, mask, _ = inputs
        x = tf.concat([img, mask], axis=-1)

        reconstruction_loss = 0.0
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions, perplexity = self(x)
            img_reconstructed, mask_reconstructed = tf.split(reconstructions, num_or_size_splits=2, axis=-1)
            
            # loss
            reconstruction_loss = tf.reduce_mean((img_reconstructed-img)**2)
            l = reconstruction_loss + self.quantizer.losses
            l = l/self.num_gpus

        grads = tape.gradient(l, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(l)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.quantize_loss_tracker.update_state(self.quantizer.losses)
        self.perplexity_tracker.update_state(perplexity)

        return {"loss": self.loss_tracker.result(),
                "reconst_loss": self.reconstruction_loss_tracker.result(),
                "quantize_loss": self.quantize_loss_tracker.result(),
                "perplexity": self.perplexity_tracker.result(),}
    
    def test_step(self, data):
        img, mask, _ = data
        x = tf.concat([img, mask], axis=-1)

        # Outputs from the VQ-VAE
        reconstructions, _ = self(x)

        img_reconstructed, mask_reconstructed = tf.split(reconstructions, num_or_size_splits=2, axis=-1)

        # Compute reconstruction loss
        reconstruction_loss = tf.reduce_mean((img - img_reconstructed)**2)
        loss = reconstruction_loss + self.quantizer.losses
        loss = loss / self.num_gpus

        # Update metrics
        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.quantize_loss_tracker.update_state(self.quantizer.losses)

        ssim_scores = tf.map_fn(
            lambda z: tf.image.ssim(z[0], z[1], max_val=tf.reduce_max(z[1]) - tf.reduce_min(z[1])),
            (img, img_reconstructed),
            dtype=tf.float32
        )
        self.ssim_tracker.update_state(tf.reduce_mean(ssim_scores))

        psnr_value = tf.map_fn(
            lambda z: tf.image.psnr(z[0], z[1], max_val=tf.reduce_max(z[1]) - tf.reduce_min(z[1])),
            (img, img_reconstructed),
            dtype=tf.float32
        )
        # psnr_value = tf.image.psnr(img, img_reconstructed, max_val=tf.reduce_max(img_reconstructed) - tf.reduce_min(img_reconstructed))
        self.psnr_tracker.update_state(psnr_value)
        # self.perplexity_tracker.update_state(perplexity)

        return {"loss": self.loss_tracker.result(), 
                "reconst_loss": self.reconstruction_loss_tracker.result(), 
                "quantize_loss": self.quantize_loss_tracker.result(),
                "ssim": self.ssim_tracker.result(),
                "psnr": self.psnr_tracker.result(),
                }

    def get_vq_model(self):
        return self.quantizer
