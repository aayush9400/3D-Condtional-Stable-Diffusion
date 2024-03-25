import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from dipy.align.reslice import reslice
import wandb


import numpy as np
import matplotlib.pyplot as plt
import wandb
import tensorflow as tf


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
            for x, _ in self.val_dataset.take(1):
                break

            # Ensure we only use a subset for visualization to save resources
            x_subset = x[:self.num_images]

            # Get the reconstructions from the VQVAE model
            reconstructions, _ = self.model(x_subset, training=False)

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
    def __init__(self, nsvq_layer, batch_size, frequency=10):
        self.nsvq_layer = nsvq_layer
        self.frequency = frequency
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.frequency == 0:
            print(f"Epoch {epoch+1}: Replacing unused codebooks.")
            self.nsvq_layer.replace_unused_codebooks(self.batch_size)


class NSVQ(keras.Model):
    def __init__(self, num_embeddings, embedding_dim, discarding_threshold=0.01, initialization='normal', **kwargs):
        super(NSVQ, self).__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.discarding_threshold = discarding_threshold
        self.eps = tf.constant(1e-12, dtype=tf.float32)

        # if initialization=='normal':
        #     w_init = tf.random_normal_initializer()
        # else:
        #     w_init = tf.keras.initializers.HeUniform()
        
        w_init = tf.keras.initializers.HeUniform()

        self.codebooks = tf.Variable(
            initial_value=w_init(
                shape=(self.num_embeddings, self.embedding_dim), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

        self.codebooks_used = tf.Variable(tf.zeros([num_embeddings], dtype=tf.int32), trainable=False, name='codebooks_used')

    def call(self, input_data, training=None):
        # tf.print(input_data.shape, self.codebooks.shape)
        input_shape = tf.shape(input_data)
        flattened = tf.reshape(input_data, [-1, self.embedding_dim])
        # compute the distances between input and codebooks vectors
        similarity = tf.matmul(flattened, self.codebooks, transpose_b=True)
        distances = (
            tf.reduce_sum(flattened ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(tf.transpose(self.codebooks) ** 2, axis=0)
            - 2 * similarity
        )

        min_indices = tf.argmin(distances, axis=1)
        # print("input", input_data.shape)
        hard_quantized_input = tf.gather(self.codebooks, min_indices)
        # print("indice", min_indices.shape)

        # Construct the new shape for hard_quantized_input
        input_shape = tf.shape(input_data)  # Dynamic shape of input_data
        new_shape = tf.concat([input_shape[:-1], [self.embedding_dim]], axis=0)
        # print(input_shape, new_shape)

        # Reshape hard_quantized_input to match the new shape
        hard_quantized_input = tf.reshape(hard_quantized_input, new_shape)
        if not training:
            quantized_input = hard_quantized_input
        else:
            batch_size = tf.shape(input_data)[0]  # Dynamically obtain the batch size
            random_vector_shape = tf.concat([[batch_size], tf.shape(input_data)[1:]], axis=0)  # Construct the shape explicitly
            random_vector = tf.random.normal(random_vector_shape) 

            norm_quantization_residual = tf.sqrt(tf.reduce_sum((input_data - hard_quantized_input) ** 2, axis=1, keepdims=True))
            norm_random_vector = tf.sqrt(tf.reduce_sum(random_vector ** 2, axis=1, keepdims=True))

            # defining vector quantization error
            vq_error = (norm_quantization_residual / (norm_random_vector + self.eps)) * random_vector

            quantized_input = input_data + vq_error
            
        # calculating the perplexity (average usage of codebook entries)
        encodings = tf.one_hot(min_indices, self.num_embeddings)
        avg_probs = tf.reduce_mean(encodings, axis=0)
        perplexity = tf.exp(-tf.reduce_sum(avg_probs * tf.math.log(avg_probs + self.eps)))

        self.codebooks_used.assign_add(tf.cast(tf.math.bincount(tf.cast(min_indices, tf.int32), minlength=self.num_embeddings), tf.int32))

        self.add_loss(tf.reduce_mean(tf.square(input_data - quantized_input)))
        # self.add_metric(self.codebooks_used, name='codebook_usage', aggregation='mean')

        return quantized_input, perplexity, self.codebooks_used

    def replace_unused_codebooks(self, num_batches):
        usage_rates = self.codebooks_used / num_batches

        unused_indices = tf.where(usage_rates < self.discarding_threshold)
        used_indices = tf.where(usage_rates >= self.discarding_threshold)

        unused_indices = tf.cast(unused_indices, tf.int64)
        # print(unused_indices.shape)
        unused_count = tf.shape(unused_indices)[0]
        used_count = tf.shape(used_indices)[0]
        print("Used: ", used_count.numpy(), "Unused:", unused_count.numpy())
        if used_count == 0:
            print(f'####### used_indices equals zero / shuffling whole codebooks ######')
            # Add a small random noise to the whole codebooks
            self.codebooks.assign_add(self.eps * tf.random.normal(self.codebooks.shape))
        else:
            used = tf.gather(tf.stop_gradient(self.codebooks), used_indices[:, 0])
            unused = tf.gather(tf.stop_gradient(self.codebooks), unused_indices[:, 0])
            # print(used.shape)
            if used_count < unused_count:
                used_codebooks = tf.tile(used, [int((unused_count.numpy() / (used_count.numpy() + self.eps)) + 1), 1])
                used_codebooks = tf.random.shuffle(used_codebooks)
                # print(used_codebooks.shape)
            else:
                used_codebooks = used
            # print(unused_indices.shape)
            unused_indices = tf.reshape(unused_indices, [-1, 1])
            zeros_updates = tf.zeros_like(unused, dtype=tf.float32)
            # print(zeros_updates.shape)
            self.codebooks.assign(tf.tensor_scatter_nd_update(self.codebooks, unused_indices, zeros_updates))

            noise = self.eps * tf.random.normal(shape=(unused_count, self.embedding_dim))
            updated_values = tf.gather(used_codebooks, range(unused_count)) + noise

            self.codebooks.assign(tf.tensor_scatter_nd_update(self.codebooks, unused_indices, updated_values))
        
        print(f'************* Replaced ' + str(unused_count) + f' codebooks *************')
        self.codebooks_used.assign(tf.zeros_like(self.codebooks_used))


class VQVAEResidualUnit(keras.Model):
    def __init__(self, input_channels, num_res_channels, act='swish'):
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
                    layers.Conv3D(self.num_channels[i], 
                                  self.downsample_parameters[i][1], 
                                  strides=self.downsample_parameters[i][0],
                                  dilation_rate=self.downsample_parameters[i][2],
                                  padding=self.downsample_parameters[i][3],
                                  input_shape=(128, 128, 128, 1)
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
                        num_res_channels=reversed_num_res_channels[i],
                        # act=self.act
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
                 num_embeddings=32,
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
        self.alpha = 1e2
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=embedding_dim,
            num_channels=num_channels,
            num_res_layers=num_res_layers,
            num_res_channels=num_res_channels,
            downsample_parameters=downsample_parameters,
            dropout=dropout,
            # act=act,
            # kernel_resize=kernel_resize
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

        self.quantizer = NSVQ(
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim)

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconst_loss")
        self.quantize_loss_tracker = keras.metrics.Mean(name="quantize_loss")
        self.perplexity_tracker = keras.metrics.Mean(name="perplexity")
        self.ssim_tracker = keras.metrics.Mean(name="SSIM")
        self.psnr_tracker = keras.metrics.Mean(name="PSNR")


    def call(self, x, training=None):
        encoded_inputs = self.encoder(x)
        quantized, perplexity, _ = self.quantizer(encoded_inputs, training)
        decoder_outputs = self.decoder(quantized)
        return decoder_outputs, perplexity

    def call_2(self, x):
        encoded_inputs = self.encoder(x)
        quantized, perplexity, _ = self.quantizer.inference(encoded_inputs)
        tf.print(quantized.shape)
        decoder_outputs = self.decoder(quantized)
        return decoder_outputs, perplexity
    
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
        #print("printing: ", x, type(x), x.shape)
        x, mask = inputs
        reconstruction_loss = 0.0
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions, perplexity = self(x, training=True)

            # loss
            reconstruction_loss = tf.reduce_mean((reconstructions - x) ** 2) / self.num_gpus
            quantize_loss = sum(self.quantizer.losses) / self.num_gpus # Incorporate quantization loss
            loss = reconstruction_loss + self.alpha * quantize_loss 

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.quantize_loss_tracker.update_state(quantize_loss)
        self.perplexity_tracker.update_state(perplexity)

        return {
                "loss": self.loss_tracker.result(),
                "reconst_loss": self.reconstruction_loss_tracker.result(),
                "quantize_loss": self.quantize_loss_tracker.result(),
                "perplexity": self.perplexity_tracker.result(),}
    
    def test_step(self, data):
        x, mask = data

        # Outputs from the VQ-VAE
        reconstructions, _ = self(x, training=False)  # Ensure the model is in inference mode

        # Compute reconstruction loss
        reconstruction_loss = tf.reduce_mean((reconstructions - x) ** 2) / self.num_gpus
        quantize_loss = sum(self.quantizer.losses) / self.num_gpus # Incorporate quantization loss
        loss = reconstruction_loss + self.alpha * quantize_loss 

        # Update metrics
        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.quantize_loss_tracker.update_state(quantize_loss)  # Update quantization loss metric
        # self.perplexity_tracker.update_state(perplexity)
        
        ssim_scores = tf.map_fn(
            lambda z: tf.image.ssim(z[0], z[1], max_val=tf.reduce_max(z[1]) - tf.reduce_min(z[1])),
            (x, reconstructions),
            dtype=tf.float32
        )
        self.ssim_tracker.update_state(tf.reduce_mean(ssim_scores))

        psnr_value = tf.map_fn(
            lambda z: tf.image.psnr(z[0], z[1], max_val=tf.reduce_max(z[1]) - tf.reduce_min(z[1])),
            (x, reconstructions),
            dtype=tf.float32
        )
        self.psnr_tracker.update_state(psnr_value)

        # sample_image_slice = reconstructions[0,:,:,64,0]
        # tf.print(sample_image_slice)
        # print("Is Eager Execution Enabled: ", tf.executing_eagerly())
        # tf.config.run_functions_eagerly(True)
        # print("Is Eager Execution Enabled Now: ", tf.executing_eagerly())
        # sample_image_slice = sample_image_slice.numpy()
        # print(type(sample_image_slice))
        # tf.config.run_functions_eagerly(False)
        # print("Is Eager Execution Enabled Now: ", tf.executing_eagerly())
        # Return a dictionary mapping metric names to current value
        return {
            "loss": self.loss_tracker.result(), 
            "reconst_loss": self.reconstruction_loss_tracker.result(), 
            "quantize_loss": self.quantize_loss_tracker.result(),
            "ssim": self.ssim_tracker.result(),
            "psnr": self.psnr_tracker.result(),
            # "sample_reconstruction": wandb.Image(sample_image_slice, caption="Reconstructed"),
        }
    
    def get_nsvq_model(self):
        return self.quantizer
