import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

# import tensorflow_probability as tfp
import tensorflow as tf
from dipy.align.reslice import reslice

from models.lpips_tensorflow import learned_perceptual_metric_model
import wandb


class EpochCounterCallback(tf.keras.callbacks.Callback):
    def __init__(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.model.epoch_counter.assign(epoch + 1)

class SIRENActivation(layers.Layer):
    def __init__(self, w0=1.0, **kwargs):
        super(SIRENActivation, self).__init__(**kwargs)
        self.w0 = w0

    def call(self, inputs):
        return tf.sin(self.w0 * inputs)


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = tf.reduce_mean(tf.nn.relu(1. - logits_real))
    loss_fake = tf.reduce_mean(tf.nn.relu(1. + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (tf.reduce_mean(tf.nn.softplus(-logits_real)) + 
                    tf.reduce_mean(tf.nn.softplus(logits_fake)))
    return d_loss


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
            for x, mask, _ in self.val_dataset.take(self.num_images):
                break
            x = tf.concat([x, mask], axis=-1)
            # Ensure we only use a subset for visualization to save resources
            x_subset = x[:self.num_images]

            # Get the reconstructions from the VQVAE model
            reconstructions, _ = self.model(x_subset, training=False)
            img_reconstructed, mask_reconstructed = tf.split(reconstructions, num_or_size_splits=2, axis=-1)
            # print(x.shape, type(x), img_reconstructed.shape, type(img_reconstructed))
            # Prepare images for logging
            images = []
            for i in range(self.num_images):
                # Extract the specific layer from the original and reconstructed images
                original_slice = x[i, :, :, self.layer_idx, 0].numpy()
                reconstructed_slice = img_reconstructed[i,:,:,self.layer_idx,0].numpy()
                # print(original_slice.shape, type(original_slice), reconstructed_slice.shape, type(reconstructed_slice))
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
        # w_init = tf.keras.initializers.HeUniform()
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.num_embeddings, self.embedding_dim), dtype="float32"
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
        quantized = tf.matmul(encodings, self.embeddings)
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((tf.stop_gradient(x) - quantized) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)

        avg_probs = tf.reduce_mean(encodings, axis=0)
        perplexity = tf.exp(-tf.reduce_sum(avg_probs * tf.math.log(avg_probs + self.eps)))

        self.codebooks_used.assign_add(tf.cast(tf.math.bincount(tf.cast(encoding_indices, tf.int32), minlength=self.num_embeddings), tf.int32))

        return quantized, perplexity

    def get_code_indices(self, flattened_inputs, distribution=False):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings, transpose_b=True)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(tf.transpose(self.embeddings) ** 2, axis=0)
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
    def __init__(self, input_channels, num_res_channels, act_fn, dropout=None):
        super().__init__()
        self.num_res_channels = num_res_channels
        self.input_channels = input_channels
        self.act_fn = act_fn
        self.dropout = dropout
        self.conv1 = layers.Conv3D(
            self.num_res_channels, 3, activation='relu', strides=1, padding='same')
        self.conv2 = tf.keras.Sequential()
        self.conv2.add(layers.Conv3D(self.input_channels,
                       3, strides=1, padding='same'))
        if input_channels == 2:
            self.conv2.add(tf.keras.layers.GroupNormalization(groups=1))
        else:
            self.conv2.add(tf.keras.layers.GroupNormalization(8, epsilon=1e-6))
        # self.conv2.add(tf.keras.layers.BatchNormalization())

        if self.dropout:
            self.conv2.add(tf.keras.layers.Dropout(self.dropout))

        if self.act_fn == 'prelu':
            self.conv2.add(tf.keras.layers.PReLU())
        elif act_fn == 'siren':
            self.conv2.add(SIRENActivation(w0=1.0))
        elif act_fn == 'swish':
            self.conv2.add(tf.keras.layers.Activation('swish'))


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

    def __init__(self, in_channels, out_channels, num_channels, num_res_layers, num_res_channels, downsample_parameters, act_fn, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_channels = num_channels
        self.num_res_layers = num_res_layers
        self.num_res_channels = num_res_channels
        self.downsample_parameters = downsample_parameters
        self.dropout = dropout
        self.act_fn = act_fn

        self.blocks = tf.keras.Sequential()
        self.blocks.add(tf.keras.Input(shape=(128, 128, 128, self.in_channels)))

        for i in range(len(self.num_channels)):
            if i == 0:
                self.blocks.add(
                    layers.Conv3D(self.num_channels[i], 
                                  self.downsample_parameters[i][1], 
                                  strides=self.downsample_parameters[i][0],
                                  dilation_rate=self.downsample_parameters[i][2],
                                  padding=self.downsample_parameters[i][3],
                                  input_shape=(128, 128, 128, self.in_channels)
                                  ))
            else:
                self.blocks.add(
                    layers.Conv3D(self.num_channels[i], 
                                  self.downsample_parameters[i][1], 
                                  strides=self.downsample_parameters[i][0],
                                  dilation_rate=self.downsample_parameters[i][2],
                                  padding=self.downsample_parameters[i][3],
                                  ))
            # self.blocks.add(tf.keras.layers.BatchNormalization())
            self.blocks.add(tf.keras.layers.GroupNormalization(8, epsilon=1e-6))

            if i > 0 and self.dropout:
                self.blocks.add(layers.Dropout(self.dropout))

            if self.act_fn == 'prelu':
                self.blocks.add(tf.keras.layers.PReLU())
            elif act_fn == 'siren':
                self.blocks.add(SIRENActivation(w0=1.0))
            elif act_fn == 'swish':
                self.blocks.add(tf.keras.layers.Activation('swish'))
            elif act_fn == 'leaky':
                self.blocks.add(tf.keras.layers.LeakyReLU())
            
            for _ in range(self.num_res_layers):
                self.blocks.add(
                    VQVAEResidualUnit(
                        input_channels=self.num_channels[i],
                        num_res_channels=self.num_res_channels[i],
                        act_fn='prelu',
                        # dropout=self.dropout,
                    )
                )

        self.blocks.add(layers.Conv3D(self.out_channels, 3, strides=1, padding='same'))

        if self.dropout:
            self.blocks.add(layers.Dropout(self.dropout))

        if self.act_fn == 'prelu':
            self.blocks.add(tf.keras.layers.PReLU())
        elif act_fn == 'siren':
            self.blocks.add(SIRENActivation(w0=1.0))
        elif act_fn == 'swish':
            self.blocks.add(tf.keras.layers.Activation('swish'))
        elif act_fn == 'leaky':
            self.blocks.add(tf.keras.layers.LeakyReLU())

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
            dropout, output_act, act_fn, kernel_resize=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_channels = num_channels
        self.num_res_layers = num_res_layers
        self.num_res_channels = num_res_channels
        self.upsample_parameters = upsample_parameters
        self.dropout = dropout
        self.act_fn = act_fn
        self.output_act = output_act
        self.ICNR = kernel_resize

        reversed_num_channels = list(reversed(self.num_channels))

        self.blocks = []

        self.blocks.append(layers.Conv3D(reversed_num_channels[0], 3, strides=1, padding='same'))
        # self.blocks.append(tf.keras.layers.BatchNormalization())
        self.blocks.append(tf.keras.layers.GroupNormalization(8, epsilon=1e-6))

        if self.dropout:
            self.blocks.append(layers.Dropout(self.dropout))

        if self.act_fn == 'prelu':
            self.blocks.append(tf.keras.layers.PReLU())
        elif act_fn == 'siren':
            self.blocks.append(SIRENActivation(w0=1.0))
        elif act_fn == 'swish':
            self.blocks.append(tf.keras.layers.Activation('swish'))

        reversed_num_res_channels = list(reversed(self.num_res_channels))
        for i in range(len(self.num_channels)):
            for _ in range(self.num_res_layers):
                self.blocks.append(
                    VQVAEResidualUnit(
                        input_channels=reversed_num_channels[i],
                        num_res_channels=reversed_num_res_channels[i],
                        act_fn='prelu',
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
            # self.blocks.append(tf.keras.layers.BatchNormalization())
            if out < 32:
                self.blocks.append(tf.keras.layers.GroupNormalization(int(out/2)))
            else:
                self.blocks.append(tf.keras.layers.GroupNormalization(8, epsilon=1e-6))

            if i != len(self.num_channels) - 1:
                if self.dropout:
                    self.blocks.append(layers.Dropout(self.dropout))

                if self.act_fn == 'prelu':
                    self.blocks.append(tf.keras.layers.PReLU())
                elif act_fn == 'siren':
                    self.blocks.append(SIRENActivation(w0=1.0))
                elif act_fn == 'swish':
                    self.blocks.append(tf.keras.layers.Activation('swish'))
                elif self.act_fn == 'relu':
                    self.blocks.append(tf.keras.layers.ReLU())

        if self.output_act:
            self.blocks.append(layers.ReLU())

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Discriminator3D(keras.Model):
    def __init__(self, in_channels, num_channels, downsample_parameters, dropout=None, getIntermFeat=True, use_sigmoid=False):
        super().__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.downsample_parameters = downsample_parameters
        self.dropout = dropout
        self.getIntermFeat = getIntermFeat
        self.use_sigmoid = use_sigmoid

        self.blocks = []

        for i in range(len(self.num_channels)):
            block = tf.keras.Sequential()
            if i == 0:
                block.add(layers.Conv3D(self.num_channels[i],
                                        kernel_size=self.downsample_parameters[i][1], 
                                        strides=2,
                                        padding='same',
                                        input_shape=(None, None, None, self.in_channels)
                                        ))
            else:
                block.add(layers.Conv3D(self.num_channels[i], 
                                        kernel_size=self.downsample_parameters[i][1], 
                                        strides=2,
                                        dilation_rate=self.downsample_parameters[i][2],
                                        padding='same',
                                        ))

            # if i > 0 and self.dropout:
            #     block.add(layers.Dropout(self.dropout))
            block.add(layers.LeakyReLU(alpha=0.2))
            self.blocks.append(block)
        
        # Remove the final convolution layer and replace with Flatten and Dense
        self.flatten = layers.Flatten()
        self.final_dense = tf.keras.models.Sequential([
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1)
                ])
        # self.final_conv = layers.Conv3D(self.num_channels[len(self.num_channels)-1], 
        #                                 kernel_size=self.downsample_parameters[i][1], 
        #                                 strides=1,
        #                                 padding='same',
        #                                 )
        if self.use_sigmoid == True:
            self.sigmoid = layers.Activation('sigmoid')

    def call(self, x):
        res = [x]
        for block in self.blocks:
            x = block(x)
            if self.getIntermFeat:
                res.append(x)

        x = self.flatten(x)
        x = self.final_dense(x)

        # x = self.final_conv(x)
        
        if self.use_sigmoid == True:
            x = self.sigmoid(x)
        
        if self.getIntermFeat:
            res.append(x)
            return x, res[1:] 
        else:
            return x, None


class Discriminator2D(keras.Model):
    def __init__(self, in_channels, num_channels, downsample_parameters, dropout=None, getIntermFeat=True, use_sigmoid=False):
        super().__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.downsample_parameters = downsample_parameters
        self.dropout = dropout
        self.getIntermFeat = getIntermFeat
        self.use_sigmoid = use_sigmoid

        self.blocks = []

        for i in range(len(self.num_channels)):
            block = tf.keras.Sequential()
            if i == 0:
                block.add(layers.Conv2D(self.num_channels[i],
                                        kernel_size=self.downsample_parameters[i][1], 
                                        strides=2,
                                        padding='same',
                                        input_shape=(None, None, self.in_channels)
                                        ))
            else:
                block.add(layers.Conv2D(self.num_channels[i], 
                                        kernel_size=self.downsample_parameters[i][1], 
                                        strides=2,
                                        dilation_rate=self.downsample_parameters[i][2],
                                        padding='same',
                                        ))

            # if i > 0 and self.dropout:
            #     block.add(layers.Dropout(self.dropout))
            block.add(layers.LeakyReLU(alpha=0.2))
            self.blocks.append(block)
        
        self.flatten = layers.Flatten()
        self.final_dense = tf.keras.models.Sequential([
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1)
                ])

        # self.final_conv = layers.Conv2D(self.num_channels[len(self.num_channels)-1], 
        #                                 kernel_size=self.downsample_parameters[i][1], 
        #                                 strides=1,
        #                                 padding='same',
        #                                 )

        if self.use_sigmoid == True:
            self.sigmoid = layers.Activation('sigmoid')

    def call(self, x):
        res = [x]
        for block in self.blocks:
            x = block(x)
            if self.getIntermFeat:
                res.append(x)

        x = self.flatten(x)
        x = self.final_dense(x)

        # x = self.final_conv(x)

        if self.use_sigmoid == True:
            x = self.sigmoid(x)
        
        if self.getIntermFeat:
            res.append(x)
            return x, res[1:]
        else:
            return x, None


class VQGAN(keras.Model):
    def __init__(self, in_channels, out_channels, num_channels, num_res_layers, num_res_channels,
                 downsample_parameters=(
                     (2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1)),
                 upsample_parameters=(
                     (2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
                 num_embeddings=128,
                 embedding_dim=64,
                 dropout=0.5,
                 output_act=None,
                 num_gpus=2,
                 kernel_resize=False,
                 B=12,
                 D=128,
                 disc_threshold=0,
                 disc_loss_fn='vanilla',
                 act_fn='prelu',
                 disc_use_sigmoid=False,
                 disc_wt=0.8,
                 lpips_wt=4,
                 gan_feat_wt=4,
                 g_loss_adv_wt=1,
                 ):

        super().__init__()
        self.B = B
        self.D = D
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_channels = num_channels
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_res_layers = num_res_layers
        self.num_res_channels = num_res_channels
        self.num_gpus = num_gpus
        if disc_loss_fn == 'vanilla':
            self.disc_loss_fn = vanilla_d_loss
        else:
            self.disc_loss_fn = hinge_d_loss
        self.epoch_counter = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.disc_threshold = disc_threshold
        self.act_fn = act_fn
        self.disc_use_sigmoid = disc_use_sigmoid
        self.lpips_wt = lpips_wt
        self.gan_feat_wt = gan_feat_wt
        self.g_loss_adv_wt = g_loss_adv_wt
        self.disc_wt = disc_wt

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=embedding_dim,
            num_channels=num_channels,
            num_res_layers=num_res_layers,
            num_res_channels=num_res_channels,
            downsample_parameters=downsample_parameters,
            # dropout=dropout,
            act_fn = 'prelu'
        )

        self.decoder = Decoder(
            in_channels=embedding_dim,
            out_channels=out_channels,
            num_channels=num_channels,
            num_res_layers=num_res_layers,
            num_res_channels=num_res_channels,
            upsample_parameters=upsample_parameters,
            dropout=dropout,
            output_act=output_act,
            kernel_resize=kernel_resize,
            act_fn = 'prelu'
        )

        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        
        self.discriminator = Discriminator3D(in_channels=self.in_channels - 1,
                                             num_channels=num_channels,
                                             downsample_parameters=downsample_parameters,
                                             dropout=dropout,
                                             use_sigmoid=disc_use_sigmoid)
        
        self.discriminator_2d = Discriminator2D(in_channels=self.in_channels - 1,
                                                num_channels=num_channels,
                                                downsample_parameters=downsample_parameters,
                                                dropout=dropout,
                                                use_sigmoid=disc_use_sigmoid)
        model_dir = './models'
        vgg_ckpt_fn = os.path.join(model_dir, 'vgg', 'exported')
        lin_ckpt_fn = os.path.join(model_dir, 'lin', 'exported')
        self.lpips = learned_perceptual_metric_model(D, vgg_ckpt_fn, lin_ckpt_fn)

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconst_loss")
        self.quantize_loss_tracker = keras.metrics.Mean(name="quantize_loss")
        self.gen_loss_tracker = keras.metrics.Mean(name="gan_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="disc_loss")
        self.perceptual_loss_tracker = keras.metrics.Mean(name="perceptual_loss")
        self.gan_feat_loss_tracker = keras.metrics.Mean(name="gan_feat_loss")
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
            self.gen_loss_tracker,
            self.disc_loss_tracker,
            self.perceptual_loss_tracker,
            self.gan_feat_loss_tracker,
            self.perplexity_tracker,
            self.ssim_tracker,
            self.psnr_tracker
        ]

    def train_step(self, inputs):
        img, mask, _ = inputs
        x = tf.concat([img, mask], axis=-1)

        reconstruction_loss = 0.0
        with tf.GradientTape() as ae_tape, tf.GradientTape() as disc_tape:
            # Outputs from the VQ-VAE.
            reconstructions, perplexity = self(x)
            img_recon, mask_reconstructed = tf.split(reconstructions, num_or_size_splits=2, axis=-1)

            # get 2d frames
            frame_idx = tf.random.uniform(shape=(self.B,), minval=30, maxval=120, dtype=tf.int32)
            batch_range = tf.range(self.B)
            indices = tf.stack([batch_range, frame_idx], axis=1)
            frames = tf.gather_nd(img, indices)
            frames_recon = tf.gather_nd(img_recon, indices)

            # calculate perceptual loss on 2d frames
            frames_lpips = tf.concat([frames, frames, frames], axis=-1)
            frames_recon_lpips = tf.concat([frames_recon, frames_recon, frames_recon], axis=-1)
            perceptual_loss = self.lpips([frames_lpips, frames_recon_lpips])

            # calculate MSE reconstruction loss
            reconstruction_loss = tf.reduce_mean((img_recon-img)**2)
            
            perform_disc_training = tf.greater_equal(self.epoch_counter, self.disc_threshold)

            def train_discriminator():
                tf.print("Training discriminator")
                real_logits, real_img_feat = self.discriminator(img, training=True)
                fake_logits, fake_img_feat = self.discriminator(img_recon, training=False)

                real_logits_2d, real_img_feat_2d = self.discriminator_2d(frames, training=True)
                fake_logits_2d, fake_img_feat_2d = self.discriminator_2d(frames_recon, training=False)

                g_loss_adv = 0.0
                g_loss_adv_3d = self.disc_loss_fn(tf.ones_like(fake_logits), fake_logits)
                g_loss_adv_2d = self.disc_loss_fn(tf.ones_like(fake_logits_2d), fake_logits_2d)
                g_loss_adv = g_loss_adv_3d + g_loss_adv_2d

                # Final GAN feature matching loss
                image_gan_feat_loss = 0.0
                video_gan_feat_loss = 0.0
                feat_weights = 4.0 / (3 + 1) 
                for i in range(len(fake_img_feat_2d) - 1):
                    # real_img_feat_2d[i].set_shape([self.B] + real_img_feat_2d[i].shape.as_list()[1:])
                    loss_2d = tf.reduce_mean((fake_img_feat_2d[i] - real_img_feat_2d[i])**2)
                    image_gan_feat_loss += feat_weights * loss_2d 
                for i in range(len(fake_img_feat) - 1):
                    # real_img_feat[i].set_shape([self.B] + real_img_feat[i].shape.as_list()[1:])
                    loss_3d = tf.reduce_mean((fake_img_feat[i] - real_img_feat[i])**2)
                    video_gan_feat_loss += feat_weights * loss_3d 
                gan_feat_loss = image_gan_feat_loss + video_gan_feat_loss

                disc_loss_real = self.disc_loss_fn(tf.ones_like(real_logits), real_logits)
                disc_loss_fake = self.disc_loss_fn(tf.zeros_like(fake_logits), fake_logits)
                disc_loss_real_2d = self.disc_loss_fn(tf.ones_like(real_logits_2d), real_logits_2d)
                disc_loss_fake_2d = self.disc_loss_fn(tf.zeros_like(fake_logits_2d), fake_logits_2d)
                disc_loss = (disc_loss_real + disc_loss_fake) + (disc_loss_real_2d + disc_loss_fake_2d)

                disc_loss = self.disc_wt * disc_loss
                loss = reconstruction_loss + self.quantizer.losses + self.lpips_wt * perceptual_loss + self.gan_feat_wt * gan_feat_loss + self.g_loss_adv_wt * g_loss_adv

                return disc_loss, loss, gan_feat_loss, g_loss_adv

            def skip_discriminator():
                tf.print("Not training discriminator yet")
                disc_loss = 0.0
                gan_feat_loss = 0.0
                g_loss_adv = 0.0
                loss = reconstruction_loss + self.quantizer.losses + self.lpips_wt * perceptual_loss
                return disc_loss, loss, gan_feat_loss, g_loss_adv

            # Use tf.cond to execute the conditional logic
            disc_loss, loss, gan_feat_loss, g_loss_adv = tf.cond(perform_disc_training, train_discriminator, skip_discriminator)

            # loss = reconstruction_loss + self.quantizer.losses + g_loss + perceptual_loss + gan_feat_loss
            loss = loss / self.num_gpus
            disc_loss = disc_loss / self.num_gpus

        grads = ae_tape.gradient(loss, self.encoder.trainable_variables + self.decoder.trainable_variables + self.quantizer.trainable_variables)
        self.vqvae_optimizer.apply_gradients(zip(grads, self.encoder.trainable_variables + self.decoder.trainable_variables + self.quantizer.trainable_variables))

        d_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables + self.discriminator_2d.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables + self.discriminator_2d.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.quantize_loss_tracker.update_state(self.quantizer.losses)
        self.gen_loss_tracker.update_state(g_loss_adv)
        self.disc_loss_tracker.update_state(disc_loss)
        self.gan_feat_loss_tracker.update_state(gan_feat_loss)
        self.perceptual_loss_tracker.update_state(perceptual_loss)
        self.perplexity_tracker.update_state(perplexity)

        return {"loss": self.loss_tracker.result(),
                "reconst_loss": self.reconstruction_loss_tracker.result(),
                "quantize_loss": self.quantize_loss_tracker.result(),
                "gen_loss": self.gen_loss_tracker.result(),
                "disc_loss": self.disc_loss_tracker.result(),
                "gen_feat_loss": self.gan_feat_loss_tracker.result(),
                "perceptual_loss": self.perceptual_loss_tracker.result(),
                "perplexity": self.perplexity_tracker.result(),}
    
    def test_step(self, data):
        img, mask, _ = data
        x = tf.concat([img, mask], axis=-1)

        # Outputs from the VQ-VAE
        reconstructions, _ = self(x, training=False)
        img_recon, mask_reconstructed = tf.split(reconstructions, num_or_size_splits=2, axis=-1)
        
        frame_idx = tf.random.uniform(shape=(self.B,), minval=30, maxval=120, dtype=tf.int32)
        batch_range = tf.range(self.B)
        indices = tf.stack([batch_range, frame_idx], axis=1)
        frames = tf.gather_nd(img, indices)
        frames_recon = tf.gather_nd(img_recon, indices)

        frames_lpips = tf.concat([frames, frames, frames], axis=-1)
        frames_recon_lpips = tf.concat([frames_recon, frames_recon, frames_recon], axis=-1)
        perceptual_loss = self.lpips([frames_lpips, frames_recon_lpips])

        reconstruction_loss = tf.reduce_mean((img_recon-img)**2)

        perform_disc_training = tf.greater_equal(self.epoch_counter, self.disc_threshold)

        def test_discriminator():
            tf.print("Training discriminator")
            real_logits, real_img_feat = self.discriminator(img)
            fake_logits, fake_img_feat = self.discriminator(img_recon)

            real_logits_2d, real_img_feat_2d = self.discriminator_2d(frames)
            fake_logits_2d, fake_img_feat_2d = self.discriminator_2d(frames_recon)

            g_loss_adv_3d = self.disc_loss_fn(tf.ones_like(fake_logits), fake_logits)
            g_loss_adv_2d = self.disc_loss_fn(tf.ones_like(fake_logits_2d), fake_logits_2d)
            g_loss_adv = g_loss_adv_3d + g_loss_adv_2d

            # Final GAN feature matching loss
            image_gan_feat_loss = 0
            video_gan_feat_loss = 0
            feat_weights = 4.0 / (3 + 1) 
            for i in range(len(fake_img_feat_2d) - 1):
                loss_2d = tf.reduce_mean((fake_img_feat_2d[i] - real_img_feat_2d[i])**2)
                image_gan_feat_loss += feat_weights * loss_2d 
            for i in range(len(fake_img_feat) - 1):
                loss_3d = tf.reduce_mean((fake_img_feat[i] - real_img_feat[i])**2)
                video_gan_feat_loss += feat_weights * loss_3d 
            gan_feat_loss = image_gan_feat_loss + video_gan_feat_loss

            disc_loss_real = self.disc_loss_fn(tf.ones_like(real_logits), real_logits)
            disc_loss_fake = self.disc_loss_fn(tf.zeros_like(fake_logits), fake_logits)
            disc_loss_real_2d = self.disc_loss_fn(tf.ones_like(real_logits_2d), real_logits_2d)
            disc_loss_fake_2d = self.disc_loss_fn(tf.zeros_like(fake_logits_2d), fake_logits_2d)
            disc_loss = (disc_loss_real + disc_loss_fake) + (disc_loss_real_2d + disc_loss_fake_2d)
            
            loss = reconstruction_loss + self.quantizer.losses + perceptual_loss + gan_feat_loss + g_loss_adv
            return disc_loss, loss, gan_feat_loss, g_loss_adv

        def skip_discriminator():
            tf.print("Not training discriminator yet")
            disc_loss = 0.0
            gan_feat_loss = 0.0
            g_loss_adv = 0.0
            loss = reconstruction_loss + self.quantizer.losses + perceptual_loss
            return disc_loss, loss, gan_feat_loss, g_loss_adv

        # Use tf.cond to execute the conditional logic
        disc_loss, loss, gan_feat_loss, g_loss_adv = tf.cond(perform_disc_training, test_discriminator, skip_discriminator)

        loss = loss / self.num_gpus 
        disc_loss = disc_loss / self.num_gpus

        # Update metrics
        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.quantize_loss_tracker.update_state(self.quantizer.losses)
        self.gen_loss_tracker.update_state(g_loss_adv)
        self.disc_loss_tracker.update_state(disc_loss)  
        self.gan_feat_loss_tracker.update_state(gan_feat_loss)
        self.perceptual_loss_tracker.update_state(perceptual_loss)

        ssim_scores = tf.map_fn(
            lambda z: tf.image.ssim(z[0], z[1], max_val=tf.reduce_max(z[1]) - tf.reduce_min(z[1])),
            (img, img_recon),
            dtype=tf.float32
        )
        self.ssim_tracker.update_state(tf.reduce_mean(ssim_scores))

        psnr_value = tf.map_fn(
            lambda z: tf.image.psnr(z[0], z[1], max_val=tf.reduce_max(z[1]) - tf.reduce_min(z[1])),
            (img, img_recon),
            dtype=tf.float32
        )
        self.psnr_tracker.update_state(psnr_value)

        return {"loss": self.loss_tracker.result(), 
                "reconst_loss": self.reconstruction_loss_tracker.result(), 
                "quantize_loss": self.quantize_loss_tracker.result(),
                "gen_loss": self.gen_loss_tracker.result(),
                "gen_feat_loss": self.gan_feat_loss_tracker.result(),
                "disc_loss": self.disc_loss_tracker.result(),
                "perceptual_loss": self.perceptual_loss_tracker.result(),
                "ssim": self.ssim_tracker.result(),
                "psnr": self.psnr_tracker.result(),
                }

    def get_vq_model(self):
        return self.quantizer
    
    def compile(self, vqvae_optimizer, discriminator_optimizer, **kwargs):
        super().compile(**kwargs)  # Pass any additional arguments to the superclass
        self.vqvae_optimizer = vqvae_optimizer
        self.discriminator_optimizer = discriminator_optimizer

