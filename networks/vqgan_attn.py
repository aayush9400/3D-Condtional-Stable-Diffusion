import os
import math
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


class VectorQuantizer(keras.Model):
    def __init__(self, num_embeddings, embedding_dim, no_random_restart=False, restart_thres=1.0, beta=0.25, **kwargs):
        super(VectorQuantizer, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.no_random_restart = no_random_restart
        self.restart_thres = restart_thres
        self._need_init = True  # Flag to check if initialization is needed
        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        # w_init = tf.keras.initializers.HeUniform()
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )
        self.codebooks_used = tf.Variable(tf.zeros([self.num_embeddings], dtype=tf.int32), trainable=False, name='codebooks_used')
        self.z_avg = tf.Variable(w_init(shape=(self.num_embeddings, embedding_dim)), trainable=False)
    
    def _tile(self, x):
        d, ew = tf.shape(x)
        if d < self.num_embeddings:
            n_repeats = (self.num_embeddings + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = tf.tile(x, [n_repeats, 1])
            x = x + tf.random.normal(tf.shape(x)) * std
        return x
    
    def _init_embeddings(self, z):
        flat_inputs = tf.reshape(z, [-1, self.embedding_dim])
        y = self._tile(flat_inputs)

        d = y.shape[0]
        indices = tf.random.shuffle(tf.range(d))[:self.num_embeddings]
        _k_rand = tf.gather(y, indices)

        self.embeddings.assign(_k_rand)
        self.z_avg.assign(_k_rand)
        self.codebooks_used.assign(tf.ones(self.num_embeddings))

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

    def call(self, x, training=False):
        if self._need_init and training:
            self._init_embeddings(x)
            self._need_init = False

        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])
        
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)

        # quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.nn.embedding_lookup(self.embeddings, encoding_indices)
        quantized = tf.reshape(quantized, input_shape)
        commitment_loss = tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((tf.stop_gradient(x) - quantized) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        if training:
            n_total = tf.reduce_sum(encodings, axis=0)
            encode_sum = tf.linalg.matmul(flattened, encodings, transpose_a=True)

            self.codebooks_used.assign(self.codebooks_used * 0.99 + n_total * 0.01)
            self.z_avg.assign(self.z_avg * 0.99 + encode_sum * 0.01)

            n = tf.reduce_sum(self.codebooks_used)
            weights = (self.codebooks_used + 1e-7) / (n + self.num_embeddings * 1e-7) * n
            encode_normalized = self.z_avg / tf.expand_dims(weights, 1)
            self.embeddings.assign(encode_normalized)

            if not self.no_random_restart:
                usage = tf.cast(self.codebooks_used >= self.restart_thres, tf.float32)
                self.embeddings.assign(self.embeddings * usage + (1 - usage) * tf.random.normal(self.embeddings.shape))

        avg_probs = tf.reduce_mean(encodings, axis=0)
        perplexity = tf.exp(-tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10)))

        return quantized, perplexity


class SamePadConv3D(layers.Layer):
    def __init__(self, out_channels, kernel_size, stride=1, bias=True, padding_type='SYMMETRIC'):
        super(SamePadConv3D, self).__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = [k - s for k, s in zip(kernel_size, stride)]
        
        self.pad_input = [[0, 0]]  
        for p in total_pad:
            self.pad_input.append([p // 2, p // 2 + p % 2]) 
        self.pad_input.append([0, 0])  
        
        self.padding_type = padding_type
        self.conv = layers.Conv3D(out_channels, 
                                  kernel_size=kernel_size, 
                                  strides=stride, 
                                  padding='valid', 
                                  use_bias=bias)

    def call(self, inputs):
        padded_inputs = tf.pad(inputs, self.pad_input, mode=self.padding_type)
        return self.conv(padded_inputs)
    

class SamePadConvTranspose3D(layers.Layer):
    def __init__(self, out_channels, kernel_size, stride=1, bias=True, padding_type='SYMMETRIC'):
        super(SamePadConvTranspose3D, self).__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3
        total_pad = [k - s for k, s in zip(kernel_size, stride)]

        self.pad_input = [[0, 0]]  
        for p in total_pad:
            self.pad_input.append([p // 2, p // 2 + p % 2]) 
        self.pad_input.append([0, 0]) 
        self.padding_type = padding_type

        self.convt = layers.Conv3DTranspose(out_channels, 
                                            kernel_size=kernel_size, 
                                            strides=stride, 
                                            padding='valid', 
                                            use_bias=bias)
        
    def call(self, inputs):
        padded_inputs = tf.pad(inputs, self.pad_input, mode=self.padding_type)
        return self.convt(padded_inputs)


def Normalize(norm_type, num_channels, num_groups=32):
    if norm_type == 'batch':
        return layers.BatchNormalization()
    elif norm_type == 'group':
        return layers.GroupNormalization(groups=min(num_groups, num_channels), axis=-1)
    else:
        raise ValueError("Unsupported normalization type") 
    
  
class VQVAEResidualUnit(layers.Layer):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group', padding_type='SYMMETRIC', num_groups=32):
        super(VQVAEResidualUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(norm_type, in_channels, num_groups=num_groups)
        self.conv1 = SamePadConv3D(self.out_channels, kernel_size=3, padding_type=padding_type)
        self.dropout = layers.Dropout(dropout)
        self.norm2 = Normalize(norm_type, self.out_channels, num_groups=num_groups)
        self.conv2 = SamePadConv3D(self.out_channels, kernel_size=3, padding_type=padding_type)

        if self.in_channels != self.out_channels or self.use_conv_shortcut:
            self.conv_shortcut = SamePadConv3D(self.out_channels, kernel_size=1, padding_type=padding_type)
        else:
            self.conv_shortcut = None

    def call(self, x):
        shortcut = x
        x = self.norm1(x)
        x = tf.nn.silu(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.norm2(x)
        x = tf.nn.silu(x)
        x = self.conv2(x)

        if self.conv_shortcut is not None:
            shortcut = self.conv_shortcut(shortcut)

        return x + shortcut


class Encoder(keras.Model):
    def __init__(self, n_hiddens, downsample, norm_type='group', padding_type='SYMMETRIC', num_groups=32):
        super(Encoder, self).__init__()

        n_times_downsample = np.array([int(np.log2(d)) for d in downsample])
        max_ds = n_times_downsample.max()
        
        self.conv_first = SamePadConv3D(n_hiddens, 
                                        kernel_size=3, 
                                        padding_type=padding_type)
        self.conv_blocks = []
        for i in range(max_ds):
            out_channels = n_hiddens * 2**(i+1)
            stride = [2 if d > 0 else 1 for d in n_times_downsample]
            block = tf.keras.Sequential([
                SamePadConv3D(out_channels, 
                              kernel_size=4, 
                              stride=stride, 
                              padding_type=padding_type),
                VQVAEResidualUnit(out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            ])
            self.conv_blocks.append(block)
            n_times_downsample -= 1

        self.final_block = tf.keras.Sequential([
            Normalize(num_channels=out_channels, norm_type=norm_type, num_groups=num_groups),
            layers.Activation('silu')
        ])

        self.out_channels = out_channels

    def call(self, inputs):
        h = self.conv_first(inputs)
        for block in self.conv_blocks:
            h = block(h)
        h = self.final_block(h)
        return h


class Decoder(keras.Model):
    def __init__(self, n_hiddens, upsample, norm_type='group', num_groups=32):
        super(Decoder, self).__init__()

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()

        in_channels = n_hiddens * 2**max_us
        self.final_block = tf.keras.Sequential([
            Normalize(norm_type, in_channels, num_groups=num_groups),
            layers.Activation('silu')
        ])

        self.conv_blocks = []
        for i in range(max_us):
            in_channels = in_channels if i == 0 else n_hiddens * (2 ** (max_us - i + 1))
            out_channels = n_hiddens * (2 ** (max_us - i))
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            conv_block = tf.keras.Sequential()
            conv_block.add(SamePadConvTranspose3D(out_channels, kernel_size=4, stride=us))
            conv_block.add(VQVAEResidualUnit(out_channels, out_channels, norm_type=norm_type, num_groups=num_groups))
            conv_block.add(VQVAEResidualUnit(out_channels, out_channels, norm_type=norm_type, num_groups=num_groups))
            self.conv_blocks.append(conv_block)
            n_times_upsample -= 1

        self.conv_last = SamePadConv3D(2, kernel_size=3, stride=1)

    def call(self, x):
        x = self.final_block(x)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = self.conv_last(x)
        return x


class Discriminator2D(keras.Model):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=tf.keras.layers.BatchNormalization, use_sigmoid=False, getIntermFeat=True):
        super(Discriminator2D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4  # Kernel size
        layers = []

        # Initial convolution layer
        layers.append(tf.keras.Sequential([
            tf.keras.layers.Conv2D(ndf, 
                                   kernel_size=kw, 
                                   strides=2, 
                                   padding='same', 
                                   input_shape=(None, None, input_nc)),
            tf.keras.layers.LeakyReLU(0.2)
        ]))

        nf = ndf
        for n in range(1, n_layers):
            nf = min(nf * 2, 512)
            layers.append(tf.keras.Sequential([
                tf.keras.layers.Conv2D(nf, 
                                       kernel_size=kw, 
                                       strides=2, 
                                       padding='same'),
                norm_layer(),
                tf.keras.layers.LeakyReLU(0.2)
            ]))

        nf = min(nf * 2, 512)
        layers.append(tf.keras.Sequential([
            tf.keras.layers.Conv2D(nf, kernel_size=kw, strides=1, padding='same'),
            norm_layer(),
            tf.keras.layers.LeakyReLU(0.2)
        ]))

        # Final convolution layer
        layers.append(tf.keras.Sequential([
            tf.keras.layers.Conv2D(1, kernel_size=kw, strides=1, padding='same')
        ]))

        if use_sigmoid:
            layers.append(tf.keras.Sequential([tf.keras.layers.Activation('sigmoid')]))

        if getIntermFeat:
            self.model_layers = layers
        else:
            self.model = tf.keras.Sequential(layers)

    def call(self, inputs):
        if self.getIntermFeat:
            result = [inputs]
            for layer in self.model_layers:
                result.append(layer(result[-1]))
            return result[-1], result[1:]
        else:
            return self.model(inputs), None


class Discriminator3D(tf.keras.Model):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=tf.keras.layers.BatchNormalization, use_sigmoid=False, getIntermFeat=True):
        super(Discriminator3D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4  # Kernel size, assume cubic kernels for simplicity
        layers = []

        # Initial convolution layer
        layers.append(tf.keras.Sequential([
            tf.keras.layers.Conv3D(ndf, 
                                   kernel_size=(kw, kw, kw), 
                                   strides=(2, 2, 2), 
                                   padding='same', 
                                   input_shape=(None, None, None, input_nc)),
            tf.keras.layers.LeakyReLU(0.2)
        ]))

        nf = ndf
        for n in range(1, n_layers):
            nf = min(nf * 2, 512)
            layers.append(tf.keras.Sequential([
                tf.keras.layers.Conv3D(nf, 
                                       kernel_size=(kw, kw, kw), 
                                       strides=(2, 2, 2), 
                                       padding='same'),
                norm_layer(),
                tf.keras.layers.LeakyReLU(0.2)
            ]))

        nf = min(nf * 2, 512)
        layers.append(tf.keras.Sequential([
            tf.keras.layers.Conv3D(nf, kernel_size=(kw, kw, kw), strides=(1, 1, 1), padding='same'),
            norm_layer(),
            tf.keras.layers.LeakyReLU(0.2)
        ]))

        # Final convolution layer
        layers.append(tf.keras.Sequential([
            tf.keras.layers.Conv3D(1, kernel_size=(kw, kw, kw), strides=(1, 1, 1), padding='same')
        ]))

        if use_sigmoid:
            layers.append(tf.keras.Sequential([tf.keras.layers.Activation('sigmoid')]))

        if getIntermFeat:
            self.model_layers = layers
        else:
            self.model = tf.keras.Sequential(layers)

    def call(self, inputs):
        if self.getIntermFeat:
            result = [inputs]
            for layer in self.model_layers:
                result.append(layer(result[-1]))
            return result[-1], result[1:]
        else:
            return self.model(inputs), None


class VQGAN(keras.Model):
    def __init__(self, in_channels, out_channels, num_channels, num_res_layers, num_res_channels,
                 downsample_parameters=(
                     (2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1)),
                 upsample_parameters=(
                     (2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
                 num_embeddings=128,
                 embedding_dim=64,
                 dropout=0.3,
                 output_act=None,
                 num_gpus=2,
                 kernel_resize=False,
                 B=12,
                 D=128,
                 disc_threshold=40,
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

        self.encoder = Encoder(n_hiddens=64, 
                               downsample=[2,2,2],
                               norm_type='group', 
                               padding_type='SYMMETRIC',
                               num_groups=32)
        self.decoder = Decoder(n_hiddens=64, 
                               upsample=[2,2,2],
                               norm_type='group', 
                               num_groups=32)
        self.pre_vq_conv = layers.Conv2D(self.embedding_dim, 1, padding='same')
        self.post_vq_conv = layers.Conv2D(self.embedding_dim, 1)

        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        
        self.discriminator = Discriminator3D(input_nc=self.in_channels - 1, 
                                             ndf=64, 
                                             n_layers=3,  
                                             use_sigmoid=False, 
                                             getIntermFeat=True,
                                             )
        
        self.discriminator_2d = Discriminator2D(input_nc=self.in_channels - 1, 
                                                ndf=64, 
                                                n_layers=3, 
                                                use_sigmoid=False, 
                                                getIntermFeat=True,
                                                )
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
        tf.print(x.shape)
        encoded_inputs = self.encoder(x)
        tf.print(encoded_inputs.shape)
        quantized, perplexity = self.quantizer(encoded_inputs)
        tf.print(quantized.shape)
        decoder_outputs = self.decoder(quantized)
        tf.print(decoder_outputs.shape)
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

            frame_idx = tf.random.uniform(shape=(self.B,), minval=30, maxval=120, dtype=tf.int32)
            batch_range = tf.range(self.B)
            indices = tf.stack([batch_range, frame_idx], axis=1)
            frames = tf.gather_nd(img, indices)
            frames_recon = tf.gather_nd(img_recon, indices)

            # reconstruction_loss = tf.reduce_mean((img_recon-img)**2)
            reconstruction_loss = tf.abs(img_recon - img)

            frames_lpips = tf.concat([frames, frames, frames], axis=-1)
            frames_recon_lpips = tf.concat([frames_recon, frames_recon, frames_recon], axis=-1)
            perceptual_loss = self.lpips([frames_lpips, frames_recon_lpips])
            
            perform_disc_training = tf.greater_equal(self.epoch_counter, self.disc_threshold)

            def train_discriminator():
                tf.print("Training discriminator")
                real_logits, real_img_feat = self.discriminator(img, training=True)
                fake_logits, fake_img_feat = self.discriminator(img_recon, training=False)

                real_logits_2d, real_img_feat_2d = self.discriminator_2d(frames, training=True)
                fake_logits_2d, fake_img_feat_2d = self.discriminator_2d(frames_recon, training=False)

                g_loss_adv = 0.0
                g_loss_adv_3d = -tf.reduce_mean(fake_logits)
                g_loss_adv_2d = -tf.reduce_mean(fake_logits_2d)
                g_loss_adv = g_loss_adv_3d + g_loss_adv_2d

                # Final GAN feature matching loss
                image_gan_feat_loss = 0.0
                video_gan_feat_loss = 0.0
                feat_weights = 4.0 / (3 + 1) 
                for i in range(len(fake_img_feat_2d) - 1):
                    loss_2d = tf.reduce_mean(tf.abs(fake_img_feat_2d[i] - real_img_feat_2d[i]))
                    image_gan_feat_loss += feat_weights * loss_2d 
                for i in range(len(fake_img_feat) - 1):
                    loss_3d = tf.reduce_mean(tf.abs(fake_img_feat[i] - real_img_feat[i]))
                    video_gan_feat_loss += feat_weights * loss_3d 
                gan_feat_loss = self.gan_feat_wt * (image_gan_feat_loss + video_gan_feat_loss)

                disc_loss_real = self.disc_loss_fn(tf.ones_like(real_logits), real_logits)
                disc_loss_fake = self.disc_loss_fn(tf.zeros_like(fake_logits), fake_logits)
                disc_loss_real_2d = self.disc_loss_fn(tf.ones_like(real_logits_2d), real_logits_2d)
                disc_loss_fake_2d = self.disc_loss_fn(tf.zeros_like(fake_logits_2d), fake_logits_2d)
                disc_loss = (disc_loss_real + disc_loss_fake) + (disc_loss_real_2d + disc_loss_fake_2d)

                disc_loss = self.disc_wt * disc_loss
                loss = reconstruction_loss + self.quantizer.losses + self.lpips_wt * perceptual_loss + gan_feat_loss + g_loss_adv * self.g_loss_adv_wt
                
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

        # reconstruction_loss = tf.reduce_mean((img_recon-img)**2)
        reconstruction_loss = tf.abs(img_recon - img)

        frames_lpips = tf.concat([frames, frames, frames], axis=-1)
        frames_recon_lpips = tf.concat([frames_recon, frames_recon, frames_recon], axis=-1)
        perceptual_loss = self.lpips([frames_lpips, frames_recon_lpips])

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
                loss_2d = tf.reduce_mean(tf.abs(fake_img_feat_2d[i] - real_img_feat_2d[i]))
                image_gan_feat_loss += feat_weights * loss_2d 
            for i in range(len(fake_img_feat) - 1):
                loss_3d = tf.reduce_mean(tf.abs(fake_img_feat[i] - real_img_feat[i]))
                video_gan_feat_loss += feat_weights * loss_3d 
            gan_feat_loss = image_gan_feat_loss + video_gan_feat_loss

            disc_loss_real = self.disc_loss_fn(tf.ones_like(real_logits), real_logits)
            disc_loss_fake = self.disc_loss_fn(tf.zeros_like(fake_logits), fake_logits)
            disc_loss_real_2d = self.disc_loss_fn(tf.ones_like(real_logits_2d), real_logits_2d)
            disc_loss_fake_2d = self.disc_loss_fn(tf.zeros_like(fake_logits_2d), fake_logits_2d)
            disc_loss = (disc_loss_real + disc_loss_fake) + (disc_loss_real_2d + disc_loss_fake_2d)
            disc_loss = disc_loss / self.num_gpus
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

