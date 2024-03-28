import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

# import tensorflow_probability as tfp
import tensorflow as tf
from dipy.align.reslice import reslice

import wandb


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
    def __init__(self, model, val_dataset, layer_idx=64, num_images=2, log_freq=10):
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


# class SamePadConv3D(layers.Layer):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='SYMMETRIC', **kwargs):
#         super(SamePadConv3D, self).__init__(**kwargs)
        
#         if isinstance(kernel_size, int):
#             kernel_size = (kernel_size,) * 3
#         if isinstance(stride, int):
#             stride = (stride,) * 3

#         # Calculate the total padding needed
#         total_pad = [k - s for k, s in zip(kernel_size, stride)]
        
#         # Padding for 'constant' and 'reflect' padding in TensorFlow needs to be specified for each dimension
#         self.pad_input = [[0, 0]]  # No padding for the batch size and channel dimensions
#         for p in total_pad:
#             self.pad_input.append([p // 2, p // 2 + p % 2])  # Pad more on one side if the total_pad is odd
#         self.pad_input.append([0, 0])  # No padding for the depth dimension
        
#         self.padding_type = padding_type

#         # TensorFlow uses 'channels_last' data format by default, hence the kernel_size and strides are specified differently
#         self.conv = layers.Conv3D(out_channels, kernel_size=kernel_size, strides=stride, padding='valid', use_bias=bias)

#     def call(self, inputs):
#         # Apply padding manually using tf.pad
#         padded_inputs = tf.pad(inputs, self.pad_input, mode=self.padding_type)
        
#         # Apply the 3D convolution on the padded inputs
#         return self.conv(padded_inputs)
    

# class SamePadConvTranspose3D(layers.Layer):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='constant', **kwargs):
#         super(SamePadConvTranspose3D, self).__init__(**kwargs)

#         if isinstance(kernel_size, int):
#             kernel_size = (kernel_size,) * 3
#         if isinstance(stride, int):
#             stride = (stride,) * 3

#         # Calculate the total padding needed
#         total_pad = [k - s for k, s in zip(kernel_size, stride)]

#         # Padding for 'constant' and 'reflect' in TensorFlow needs to be specified for each dimension
#         self.pad_input = [[0, 0]]  # No padding for the batch size and channel dimensions
#         for p in total_pad:
#             self.pad_input.append([p // 2, p // 2 + p % 2])  # Pad more on one side if the total_pad is odd
#         self.pad_input.append([0, 0])  # No padding for the depth dimension

#         self.padding_type = padding_type

#         # Set up the 3D transposed convolution layer
#         # Note: TensorFlow might handle padding differently for transposed convolutions, so manual padding before the layer might still be necessary
#         self.convt = layers.Conv3DTranspose(out_channels, kernel_size=kernel_size, strides=stride, padding='valid', use_bias=bias)

#     def call(self, inputs):
#         # Apply padding manually using tf.pad
#         if self.padding_type in ['constant', 'reflect']:
#             padded_inputs = tf.pad(inputs, self.pad_input, mode=self.padding_type)
#         else:
#             raise ValueError(f"Unsupported padding type: {self.padding_type}. Only 'constant' and 'reflect' are supported.")

#         # Apply the 3D transposed convolution on the padded inputs
#         return self.convt(padded_inputs)


class Discriminator3D(keras.Model):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels


        self.blocks = keras.Sequential([
            layers.Conv3D(64, kernel_size=4, strides=2, padding="same", input_shape=(128,128,128,self.in_channels)),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv3D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Flatten(),
            layers.Dense(1)
        ])

    def call(self, x):
        return self.blocks(x)


# class NLayerDiscriminator(keras.Model):
#     def __init__(self, input_nc, ndf=64, n_layers=3, norm_type='batch', use_sigmoid=False, getIntermFeat=True, **kwargs):
#         super(NLayerDiscriminator, self).__init__(**kwargs)
#         self.getIntermFeat = getIntermFeat
#         self.n_layers = n_layers

#         kw = 4
#         padw = int(np.ceil((kw - 1.0) / 2))
        
#         # Initial convolutional layer
#         sequence = [layers.Conv2D(ndf, kernel_size=kw, strides=2, padding='same', use_bias=False, input_shape=(None, None, input_nc)),
#                     layers.LeakyReLU(0.2)]

#         nf = ndf
#         for n in range(1, n_layers):
#             nf_prev = nf
#             nf = min(nf * 2, 512)
#             sequence += [
#                 layers.Conv2D(nf, kernel_size=kw, strides=2, padding='same', use_bias=False),
#                 layers.BatchNormalization(),
#                 layers.LeakyReLU(0.2)
#             ]

#         nf_prev = nf
#         nf = min(nf * 2, 512)
#         sequence += [
#             layers.Conv2D(nf, kernel_size=kw, strides=1, padding='same', use_bias=False),
#             layers.BatchNormalization(),
#             layers.LeakyReLU(0.2)
#         ]

#         sequence += [layers.Conv2D(1, kernel_size=kw, strides=1, padding='same')]

#         if use_sigmoid:
#             sequence += [layers.Activation('sigmoid')]

#         if getIntermFeat:
#             self.model_sequences = []
#             print(sequence)
#             for n in range(len(sequence)):
#                 self.model_sequences.append(tf.keras.Sequential(sequence[:n+1]))
#         else:
#             self.model = tf.keras.Sequential(sequence)

#     def call(self, inputs, training=None, mask=None):
#         if self.getIntermFeat:
#             res = [inputs]
#             for model_sequence in self.model_sequences:
#                 res.append(model_sequence(res[-1]))
#             return res[-1], res[1:]
#         else:
#             return self.model(inputs),


# class NLayerDiscriminator3D(keras.Model):
#     def __init__(self, input_nc, ndf=64, n_layers=3, norm_type='batch', use_sigmoid=False, getIntermFeat=True, **kwargs):
#         super(NLayerDiscriminator3D, self).__init__(**kwargs)
#         self.getIntermFeat = getIntermFeat
#         self.n_layers = n_layers

#         kw = 4
#         padw = int(np.ceil((kw-1.0) / 2))

#         # Define the first convolutional layer
#         sequence = [layers.Conv3D(ndf, kernel_size=kw, strides=2, padding='same', use_bias=False, input_shape=(None, None, None, input_nc)),
#                     layers.LeakyReLU(0.2)]

#         nf = ndf
#         for n in range(1, n_layers):
#             nf_prev = nf
#             nf = min(nf * 2, 512)
#             sequence += [
#                 layers.Conv3D(nf, kernel_size=kw, strides=2, padding='same', use_bias=False),
#                 layers.BatchNormalization(),
#                 layers.LeakyReLU(0.2)
#             ]

#         nf_prev = nf
#         nf = min(nf * 2, 512)
#         sequence += [
#             layers.Conv3D(nf, kernel_size=kw, strides=1, padding='same', use_bias=False),
#             layers.BatchNormalization(),
#             layers.LeakyReLU(0.2)
#         ]

#         sequence += [layers.Conv3D(1, kernel_size=kw, strides=1, padding='same')]

#         if use_sigmoid:
#             sequence += [layers.Activation('sigmoid')]

#         # Handling intermediate feature extraction
#         if getIntermFeat:
#             self.model_sequences = []
#             for n in range(len(sequence)):
#                 self.model_sequences.append(tf.keras.Sequential(sequence[:n+1]))
#         else:
#             self.model = tf.keras.Sequential(sequence)

#     def call(self, inputs, training=None, mask=None):
#         if self.getIntermFeat:
#             res = [inputs]
#             for model_sequence in self.model_sequences:
#                 res.append(model_sequence(res[-1]))
#             return res[-1], res[1:]
#         else:
#             return self.model(inputs),


class VQGAN(keras.Model):
    def __init__(self, in_channels, out_channels, num_channels, num_res_layers, num_res_channels,
                 downsample_parameters=(
                     (2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1)),
                 upsample_parameters=(
                     (2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
                 num_embeddings=128,
                 embedding_dim=64,
                 dropout=0.1,
                 output_act=None,
                 num_gpus=2,
                 kernel_resize=False,
                #  gan_feat_weight=1,
                #  image_gan_weight=1,
                #  video_gan_weight=1,
                 ):

        super().__init__()
        # self.B = batch_size
        # self.D = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_channels = num_channels
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_res_layers = num_res_layers
        self.num_res_channels = num_res_channels
        self.num_gpus = num_gpus
        # self.gan_feat_weight = gan_feat_weight
        # self.image_gan_weight = image_gan_weight
        # self.video_gan_weight = video_gan_weight

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=embedding_dim,
            num_channels=num_channels,
            num_res_layers=num_res_layers,
            num_res_channels=num_res_channels,
            downsample_parameters=downsample_parameters,
            dropout=dropout,
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
            kernel_resize=kernel_resize
        )

        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        
        self.discriminator = Discriminator3D(in_channels=self.in_channels - 1)

        # self.image_discriminator = NLayerDiscriminator(self.out_channels-1)
        # self.video_discriminator = NLayerDiscriminator3D(self.out_channels-1)

        self.disc_loss_fn = vanilla_d_loss

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconst_loss")
        self.quantize_loss_tracker = keras.metrics.Mean(name="quantize_loss")
        self.gen_loss_tracker = keras.metrics.Mean(name="gan_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="disc_loss")
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
            self.perplexity_tracker,
            self.ssim_tracker,
            self.psnr_tracker
        ]

    def train_step(self, inputs):
        img, mask, _ = inputs
        x = tf.concat([img, mask], axis=-1)

        reconstruction_loss = 0.0
        image_gan_feat_loss = 0.0
        video_gan_feat_loss = 0.0
        with tf.GradientTape() as ae_tape, tf.GradientTape() as disc_tape:
            # Outputs from the VQ-VAE.
            reconstructions, perplexity = self(x)
            img_recon, mask_reconstructed = tf.split(reconstructions, num_or_size_splits=2, axis=-1)
            
            real_logits = self.discriminator(img, training=True)
            fake_logits = self.discriminator(img_recon, training=True)

            g_loss_adv = self.disc_loss_fn(tf.ones_like(fake_logits), fake_logits)
            g_loss_recon = tf.reduce_mean((img - img_recon)**2)
            g_loss = g_loss_adv + g_loss_recon

            disc_loss_real = self.disc_loss_fn(tf.ones_like(real_logits), real_logits)
            disc_loss_fake = self.disc_loss_fn(tf.zeros_like(fake_logits), fake_logits)
            disc_loss = disc_loss_real + disc_loss_fake

            reconstruction_loss = tf.reduce_mean((img_recon-img)**2)
            l = reconstruction_loss + self.quantizer.losses + g_loss
            l = l/self.num_gpus   

        grads = ae_tape.gradient(l, self.trainable_variables)
        self.vqvae_optimizer.apply_gradients(zip(grads, self.trainable_variables))

        d_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        self.loss_tracker.update_state(l)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.quantize_loss_tracker.update_state(self.quantizer.losses)
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(disc_loss)
        self.perplexity_tracker.update_state(perplexity)

        return {"loss": self.loss_tracker.result(),
                "reconst_loss": self.reconstruction_loss_tracker.result(),
                "quantize_loss": self.quantize_loss_tracker.result(),
                "gen_loss": self.gen_loss_tracker.result(),
                "disc_loss": self.disc_loss_tracker.result(),
                "perplexity": self.perplexity_tracker.result(),}
    
    def test_step(self, data):
        img, mask, _ = data
        x = tf.concat([img, mask], axis=-1)

        # Outputs from the VQ-VAE
        reconstructions, _ = self(x, training=False)
        img_reconstructed, mask_reconstructed = tf.split(reconstructions, num_or_size_splits=2, axis=-1)

        # Compute reconstruction loss
        reconstruction_loss = tf.reduce_mean((img - img_reconstructed)**2)
        loss = reconstruction_loss + self.quantizer.losses
        loss = loss / self.num_gpus

        # Compute discriminator loss for validation
        real_logits = self.discriminator(img, training=False)  
        fake_logits = self.discriminator(img_reconstructed, training=False)

        g_loss_adv = self.disc_loss_fn(tf.ones_like(fake_logits), fake_logits)
        g_loss_recon = tf.reduce_mean((img - img_reconstructed)**2)
        g_loss = g_loss_adv + g_loss_recon

        disc_loss_real = self.disc_loss_fn(tf.ones_like(real_logits), real_logits)
        disc_loss_fake = self.disc_loss_fn(tf.zeros_like(fake_logits), fake_logits)
        val_disc_loss = (disc_loss_real + disc_loss_fake) / 2 

        # Update metrics
        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.quantize_loss_tracker.update_state(self.quantizer.losses)
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(val_disc_loss)  

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
        self.psnr_tracker.update_state(psnr_value)

        return {"loss": self.loss_tracker.result(), 
                "reconst_loss": self.reconstruction_loss_tracker.result(), 
                "quantize_loss": self.quantize_loss_tracker.result(),
                "gen_loss": self.gen_loss_tracker.result(),
                "disc_loss": self.disc_loss_tracker.result(),  # Return the validation discriminator loss
                "ssim": self.ssim_tracker.result(),
                "psnr": self.psnr_tracker.result(),
                }

    def get_vq_model(self):
        return self.quantizer
    
    def compile(self, vqvae_optimizer, discriminator_optimizer, **kwargs):
        super().compile(**kwargs)  # Pass any additional arguments to the superclass
        self.vqvae_optimizer = vqvae_optimizer
        self.discriminator_optimizer = discriminator_optimizer


# def Normalize(norm_type='group', num_groups=32):
#     assert norm_type in ['group', 'batch']
#     if norm_type == 'group':
#         # TensorFlow does not have a built-in GroupNorm, but you can use a third-party implementation
#         # or approximate it using existing layers (not directly equivalent).
#         # For demonstration purposes, this code uses BatchNormalization as a placeholder.
#         # You might want to replace this with an actual GroupNorm implementation.
#         return layers.GroupNormalization(groups=num_groups, axis=-1, epsilon=1e-6)
#     elif norm_type == 'batch':
#         return layers.BatchNormalization(axis=-1)


# class ResBlock(layers.Layer):
#     def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group', padding_type='constant', num_groups=32, **kwargs):
#         super(ResBlock, self).__init__(**kwargs)
#         self.in_channels = in_channels
#         self.out_channels = out_channels if out_channels is not None else in_channels
#         self.use_conv_shortcut = conv_shortcut

#         # Normalization layer 1
#         self.norm1 = Normalize(norm_type, num_groups)
#         # Convolution layer 1
#         self.conv1 = SamePadConv3D(in_channels, self.out_channels, kernel_size=3, padding_type=padding_type)
#         # Optional dropout layer
#         self.dropout = layers.Dropout(dropout)
#         # Normalization layer 2
#         self.norm2 = self._get_norm_layer(self.out_channels, norm_type, num_groups)
#         # Convolution layer 2
#         self.conv2 = SamePadConv3D(self.out_channels, self.out_channels, kernel_size=3, padding_type=padding_type)
#         # Convolution shortcut (if needed)
#         if self.in_channels != self.out_channels or self.use_conv_shortcut:
#             self.conv_shortcut = SamePadConv3D(in_channels, self.out_channels, kernel_size=1, padding_type=padding_type)

#     def call(self, inputs):
#         h = inputs
#         h = self.norm1(h)
#         h = tf.nn.silu(h)  # SiLU activation (Swish)
#         h = self.conv1(h)
#         h = self.dropout(h)
#         h = self.norm2(h)
#         h = tf.nn.silu(h)
#         h = self.conv2(h)

#         if self.in_channels != self.out_channels or self.use_conv_shortcut:
#             inputs = self.conv_shortcut(inputs)

#         return inputs + h


# class Encoder(keras.Model):
#     def __init__(self, n_hiddens, downsample, image_channel=3, norm_type='group', padding_type='constant', num_groups=32, **kwargs):
#         super(Encoder, self).__init__(**kwargs)
#         n_times_downsample = np.array([int(np.log2(d)) for d in downsample])
#         max_ds = n_times_downsample.max()

#         self.conv_first = SamePadConv3D(image_channel, n_hiddens, kernel_size=3, padding_type=padding_type)

#         self.conv_blocks = []
#         for i in range(max_ds):
#             in_channels = n_hiddens * 2**i
#             out_channels = n_hiddens * 2**(i+1)
#             stride = [2 if d > 0 else 1 for d in n_times_downsample]
#             block = tf.keras.Sequential([
#                 SamePadConv3D(in_channels, out_channels, kernel_size=4, stride=stride, padding_type=padding_type),
#                 ResBlock(out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
#             ])
#             self.conv_blocks.append(block)
#             n_times_downsample -= 1

#         self.final_block = tf.keras.Sequential([
#             Normalize(out_channels, norm_type=norm_type, num_groups=num_groups),
#             layers.Activation('silu')
#         ])

#         self.out_channels = out_channels

#     def call(self, inputs):
#         h = self.conv_first(inputs)
#         for block in self.conv_blocks:
#             h = block(h)
#         h = self.final_block(h)
#         return h
    

# class Decoder(keras.Model):
#     def __init__(self, n_hiddens, upsample, image_channel, norm_type='group', num_groups=32, **kwargs):
#         super(Decoder, self).__init__(**kwargs)

#         n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
#         max_us = n_times_upsample.max()

#         in_channels = n_hiddens * 2**max_us
#         self.final_block = tf.keras.Sequential([
#             Normalize(in_channels, norm_type, num_groups=num_groups),
#             layers.Activation('silu')
#         ])

#         self.conv_blocks = []
#         for i in range(max_us):
#             in_channels = in_channels if i == 0 else n_hiddens * 2**(max_us - i + 1)
#             out_channels = n_hiddens * 2**(max_us - i)
#             us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
#             conv_block = tf.keras.Sequential()
#             conv_block.add(SamePadConvTranspose3D(in_channels, out_channels, kernel_size=4, stride=us))
#             conv_block.add(ResBlock(out_channels, out_channels, norm_type=norm_type, num_groups=num_groups))
#             conv_block.add(ResBlock(out_channels, out_channels, norm_type=norm_type, num_groups=num_groups))
#             self.conv_blocks.append(conv_block)
#             n_times_upsample -= 1

#         self.conv_last = SamePadConv3D(out_channels, image_channel, kernel_size=3)

#     def call(self, inputs):
#         h = self.final_block(inputs)
#         for conv_block in self.conv_blocks:
#             h = conv_block(h)
#         h = self.conv_last(h)
#         return h
