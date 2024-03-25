import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from PIL import Image
import wandb

import tensorflow as tf
from tensorflow.keras.utils import plot_model as plot
from tensorflow import keras, einsum
from tensorflow.keras import Model, Sequential
from tensorflow.keras import layers
from networks.vqvae3d_monai import VQVAE


def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )


class WandbImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, log_freq=10, img_shape=(1, 64//8, 64//8, 64//8, 256)):
        # Initialize with the diffusion model and image shape
        self.diffusion_model = model
        self.img_shape = img_shape
        self.slice_index = 64
        self.log_freq = log_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_freq == 0:
            for context_value in range(0,2):
                for i in [self.diffusion_model.timesteps]:
                    print(f"Generating for {i} rsteps")
                    # self.diffusion_model.vqvae_trainer.load_weights(self.diffusion_model.vqvae_load_ckpt)
                    img_latents = self.diffusion_model.generate(
                        self.img_shape, last_step=self.diffusion_model.timesteps - i, context_value=context_value
                    )

                    images = self.diffusion_model.vqvae_trainer.decoder(img_latents)
                    selected_slice = images[:, :, :, self.slice_index, 0].numpy()

                    # Plot the selected slice
                    fig, ax = plt.subplots()
                    ax.imshow(selected_slice[0], cmap='gray')  # Display the first image in the batch for simplicity
                    ax.axis('off')  # Hide axes ticks

                    # Convert the plot to an image
                    fig.canvas.draw()
                    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                    plt.close(fig)  # Close the plot to free resources

                    # Log the image to WandB
                    wandb.log({f"Generated Image {context_value}": [wandb.Image(img_array, caption=f"Epoch {epoch+1}")]}, commit=False)
                

            # wandb.log({"Generated Images": [wandb.Image(images.numpy(), caption=f"Generated Image {context_value}")]}, commit=False)


class AttentionBlock(layers.Layer):
    """Applies self-attention.

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=8, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)

        # self.norm = layers.GroupNormalization(groups=groups)
        self.norm = layers.BatchNormalization()
        self.query = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.key = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.value = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.depth = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.proj = layers.Dense(units, kernel_initializer=kernel_init(0.0))

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        depth = tf.shape(inputs)[3]
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attn_score = tf.einsum("bhwdc, bHWDc->bhwdHWD", q, k) * scale
        attn_score = tf.reshape(
            attn_score, [batch_size, height, width, depth, height * width * depth]
        )

        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(
            attn_score, [batch_size, height, width, depth, height, width, depth]
        )

        proj = tf.einsum("bhwdHWD,bHWDc->bhwdc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj


class CrossAttentionBlock(layers.Layer):
    """Applies self-attention.

    Args:
        units: Number of units in the dense layers
        num_heads: Number of attention heads
    """

    def __init__(self, units, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.norm = layers.BatchNormalization()
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.norm3 = layers.LayerNormalization()
        self.relu = layers.ReLU()
        self.proj_in = layers.Conv3D(units, 1)
        self.proj_out = layers.Conv3D(units, 1)
        self.query = layers.Dense(units)
        self.key = layers.Dense(units)
        self.value = layers.Dense(units)
        self.proj = keras.Sequential([
            layers.Dense(units * 4),
            layers.ReLU(),
            layers.Dense(units),
        ])

    def reshape_head_to_batch_dim(self, inputs):
        dynamic_shape = tf.shape(inputs)
        batch_size, height, width, depth, channels = dynamic_shape[0], dynamic_shape[1], dynamic_shape[2], dynamic_shape[3], inputs.shape[4]
        seq_len = height * width * depth
        inputs = tf.reshape(inputs, [batch_size, seq_len, channels])

        if self.num_heads > 1:
            inputs = tf.reshape(inputs, [batch_size, seq_len, self.num_heads, channels // self.num_heads])
            inputs = tf.transpose(inputs, [0, 2, 1, 3])
            inputs = tf.reshape(inputs, [batch_size * self.num_heads, seq_len, channels // self.num_heads])
        return inputs

    def reshape_batch_dim_to_head(self, inputs):
        dynamic_shape = tf.shape(inputs)
        batch_size, seq_len, channels = dynamic_shape[0], dynamic_shape[1], inputs.shape[2]

        if self.num_heads > 1:
            inputs = tf.reshape(inputs, [batch_size // self.num_heads, self.num_heads, seq_len, channels])
            inputs = tf.transpose(inputs, [0, 2, 1, 3])
            inputs = tf.reshape(inputs, [batch_size // self.num_heads, seq_len, channels * self.num_heads])
        return inputs

    def attention(self, inputs, context=None):
        scale = tf.cast(self.units, tf.float32) ** (-0.5)
        dynamic_shape = tf.shape(inputs)
        batch_size, height, width, depth = dynamic_shape[0], dynamic_shape[1], dynamic_shape[2], dynamic_shape[3]

        q = self.query(inputs)
        k = self.key(inputs) if context is None else self.key(context)
        v = self.value(inputs) if context is None else self.value(context)

        q, k, v = (
            self.reshape_head_to_batch_dim(q),
            self.reshape_head_to_batch_dim(k),
            self.reshape_head_to_batch_dim(v),
        )

        attn_score = tf.einsum("blc,bLc->blL", q, k) * scale
        attn_score = tf.nn.softmax(attn_score, -1)

        proj = tf.einsum("blL,bLc->blc", attn_score, v)
        proj = self.reshape_batch_dim_to_head(proj)
        proj = tf.reshape(proj, [batch_size, height, width, depth, self.units])

        return proj

    def call(self, inputs, context=None):
        residual = inputs
        inputs = self.norm(inputs)
        inputs = self.relu(self.proj_in(inputs))

        x = self.attention(self.norm1(inputs)) + inputs
        x = self.attention(self.norm2(inputs), context) + x
        x = self.proj(self.norm3(inputs)) + x

        return self.relu(self.proj_out(x)) + residual


class TimeEmbedding(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2
        self.max_period = 10000
        self.emb = math.log(self.max_period) / (self.half_dim - 1)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)
        # freqs = tf.exp(-math.log(max_period) * tf.arange(start=0, end=self.half_dim, dtype=tf.float32) / self.half_dim)

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb


class Betas:
    def __init__(self, timesteps):
        beta = np.linspace(0.0001, 0.02, timesteps)
        alpha = 1 - beta
        sqrt_alpha = np.sqrt(alpha)
        alpha_bar = np.cumprod(alpha, 0)
        alpha_bar_prev = np.append(1.0, alpha_bar[:-1])
        sqrt_alpha_bar = np.sqrt(alpha_bar)
        sqrt_alpha_bar_prev = np.sqrt(alpha_bar_prev)
        sqrt_one_minus_alpha_bar = np.sqrt(1 - alpha_bar)

        self.beta = tf.constant(beta, dtype=tf.float32)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.sqrt_alpha = tf.constant(sqrt_alpha, dtype=tf.float32)
        self.alpha_bar = tf.constant(alpha_bar, dtype=tf.float32)
        self.alpha_bar_prev = tf.constant(alpha_bar_prev, dtype=tf.float32)
        self.sqrt_alpha_bar = tf.constant(sqrt_alpha_bar, dtype=tf.float32)
        self.sqrt_alpha_bar_prev = tf.constant(sqrt_alpha_bar_prev, dtype=tf.float32)
        self.sqrt_one_minus_alpha_bar = tf.constant(
            sqrt_one_minus_alpha_bar, dtype=tf.float32
        )


def ResidualBlock(width, groups=8, activation_fn=keras.activations.swish):
    def apply(inputs):
        x, t = inputs
        input_width = x.shape[4]

        if input_width == width:
            residual = x
        else:
            residual = layers.Conv3D(
                width, kernel_size=1, kernel_initializer=kernel_init(1.0)
            )(x)

        temb = activation_fn(t)
        temb = layers.Dense(width, kernel_initializer=kernel_init(1.0))(temb)[
            :, None, None, None, :
        ]
        # x = layers.GroupNormalization(groups=groups)(x)
        x = layers.BatchNormalization()(x)
        x = activation_fn(x)
        x = layers.Conv3D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)
        x = layers.Add()([x, temb])
        # x = layers.GroupNormalization(groups=groups)(x)
        x = layers.BatchNormalization()(x)
        x = activation_fn(x)

        x = layers.Conv3D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0)
        )(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownSample(width):
    def apply(x):
        x = layers.Conv3D(
            width,
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_initializer=kernel_init(1.0),
        )(x)
        return x

    return apply


def UpSample(width, interpolation="nearest"):
    def apply(x):
        x = layers.UpSampling3D(size=2)(x)
        x = layers.Conv3D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)
        return x

    return apply


def TimeMLP(units, activation_fn=keras.activations.swish):
    def apply(inputs):
        temb = layers.Dense(
            units, activation=activation_fn, kernel_initializer=kernel_init(1.0)
        )(inputs)
        temb = layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
        return temb

    return apply


def ContextMLP(units, activation_fn=keras.activations.swish):
    def apply(inputs):
        # print(inputs.shape)
        cemb = layers.Dense(
            units[0]*units[1]*units[2]*units[3], activation=activation_fn)(inputs)
        cemb = tf.reshape(cemb, [-1, units[0], units[1], units[2], units[3]])
        return cemb

    return apply


first_conv_channels = 32


def build_model(
    img_size,
    img_channels,
    widths,
    has_attention,
    has_cross_attention=None,
    num_res_blocks=2,
    norm_groups=8,
    interpolation="nearest",
    activation_fn=keras.activations.swish,
    context_dim=1
):

    image_input = layers.Input(
        shape=(img_size, img_size, img_size, img_channels), name="image_input"
    )
    time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")
    context_input = keras.Input(shape=(context_dim, context_dim), dtype=tf.int64, name="context_input")

    if has_cross_attention and not context_dim:
        raise ValueError(
            "Context dim can not be None if has_cross_attention is not None"
        )

    x = layers.Conv3D(
        first_conv_channels,
        kernel_size=(3, 3, 3),
        padding="same",
        kernel_initializer=kernel_init(1.0),
    )(image_input)

    temb = TimeEmbedding(dim=first_conv_channels * 4)(time_input)
    temb = TimeMLP(units=first_conv_channels * 4, activation_fn=activation_fn)(temb)
    
    cemb = layers.Embedding(context_dim, first_conv_channels * 4)(context_input)
    # cemb.trainable = False
    # x = tf.concat([x, down_cemb], axis=-1)
    skips = [x]
    # DownBlock
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = ResidualBlock(
                widths[i], groups=norm_groups, activation_fn=activation_fn
            )([x, temb])
            if has_attention[i]:
            #     x = AttentionBlock(widths[i], groups=norm_groups)(x)
            # elif has_cross_attention[i]:
                input_shape = x.shape
                down_cemb = ContextMLP(units=(input_shape[1],input_shape[2],input_shape[3],input_shape[4]), activation_fn=activation_fn)(cemb)
                x=CrossAttentionBlock(widths[i], num_heads=1)(x,down_cemb)
            skips.append(x)

        # Avoid downsampling if it's the last layer
        if widths[i] != widths[-1]:
            x = DownSample(widths[i])(x)
            skips.append(x)
    # x = tf.concat([x, middle_cemb], axis=-1)
    # MiddleBlock
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)(
        [x, temb]
    )
    # x = AttentionBlock(widths[-1], groups=norm_groups)(x)
    input_shape = x.shape
    middle_cemb = ContextMLP(units=(input_shape[1],input_shape[2],input_shape[3],input_shape[4]), activation_fn=activation_fn)(cemb)
    x = CrossAttentionBlock(widths[-1], num_heads=1)(x,middle_cemb)
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)(
        [x, temb]
    )
    # x = tf.concat([x, up_cemb], axis=-1)
    # UpBlock
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock(
                widths[i], groups=norm_groups, activation_fn=activation_fn
            )([x, temb])
            if has_attention[i]:
                # x = AttentionBlock(widths[i], groups=norm_groups)(x)
                input_shape = x.shape
                up_cemb = ContextMLP(units=(input_shape[1],input_shape[2],input_shape[3],input_shape[4]), activation_fn=activation_fn)(cemb)
                x = CrossAttentionBlock(widths[i], num_heads=1)(x,up_cemb)

        if i != 0:
            x = UpSample(widths[i], interpolation=interpolation)(x)
    # End block
    # x = layers.GroupNormalization(groups=norm_groups)(x)
    x = layers.BatchNormalization()(x)
    x = activation_fn(x)
    x = layers.Conv3D(
        img_channels, (3, 3, 3), padding="same", kernel_initializer=kernel_init(0.0)
    )(x)
    return keras.Model([image_input, time_input, context_input], x, name="unet")


class DiffusionModel(keras.Model):

    def __init__(self, latent_size, num_embed, latent_channels, vqvae_load_ckpt, args):
        super().__init__()
        self.timesteps = args.timesteps
        self.b = Betas(args.timesteps)
        self.lc = latent_channels
        self.vqvae_trainer = VQVAE(
                in_channels=1,
                out_channels=1,
                num_channels=(32, 64, 128,256),
                num_res_channels=(32, 64, 128,256),
                num_res_layers=5,
                # downsample_parameters=(stride, kernel_size, dilation_rate, padding)
                downsample_parameters=(
                    (2, 4, 1, "same"),
                    (2, 4, 1, "same"),
                    (2, 4, 1, "same"),
					(2, 4, 1, "same"),
                ),
                upsample_parameters=(
                    (2, 4, 1, "same", 0),
                    (2, 4, 1, "same", 0),
                    (2, 4, 1, "same", 0),
					(2, 4, 1, "same", 0),
                ),
                num_embeddings=num_embed,
                embedding_dim=latent_channels,
                dropout=None,
                num_gpus=args.num_gpus,
                kernel_resize=args.kernel_resize,
            )

        if vqvae_load_ckpt is not None:
            print("Loading VQVAE weights")
            self.vqvae_load_ckpt = vqvae_load_ckpt
            self.vqvae_trainer.load_weights(vqvae_load_ckpt)
        self.encoder = self.vqvae_trainer.encoder
        self.quantizer = self.vqvae_trainer.quantizer
        self.decoder = self.vqvae_trainer.decoder
        self.quantizer.trainable = False
        self.encoder.trainable = False
        self.decoder.trainable = False
        self.network = build_model(
            latent_size,
            latent_channels,
            widths=[64, 128, 256],
            has_attention=[False, False, True, True],
        )
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.num_gpus = args.num_gpus
        self.global_bs = args.bs

    def train_step(self, inputs):
        images, _, context = inputs
        batch_size = tf.shape(images)[0]
        t = tf.random.uniform(
            minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64
        )
        with tf.GradientTape() as tape:
            latents, _ = self.quantizer(self.encoder(images))

            # 3. Sample random noise to be added to the images in the batch
            noise = tf.random.normal(shape=tf.shape(latents), dtype=images.dtype)

            # 4. Diffuse the images with noise
            sqb = tf.reshape(
                tf.gather(self.b.sqrt_alpha_bar, t), [batch_size, 1, 1, 1, 1]
            )
            osqb = tf.reshape(
                tf.gather(self.b.sqrt_one_minus_alpha_bar, t), [batch_size, 1, 1, 1, 1]
            )
            noisy_img = sqb * latents + osqb * noise

            # 5. Pass the diffused images and time steps to the network
            pred_noise = self.network([noisy_img, t, context], training=True)

            # 6. Calculate the loss
            loss_reduction_factor = (
                self.global_bs * self.lc * self.lc * self.lc * self.lc * 1.0
            )
            lo = self.loss(noise, pred_noise) / loss_reduction_factor  # /12800.0

        gradients = tape.gradient(lo, self.network.trainable_weights)

        # 8. Update the weights of the network
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        # 9. Update metrics
        self.loss_tracker.update_state(lo)

        # return {"loss": lo}
        return {"loss": self.loss_tracker.result()}

    # Tracks loss metric and resets after every epoch
    @property
    def metrics(self):
        return [self.loss_tracker]

    def sample(self, x_t, pred_noise, curr_time_step, shape):
        b = tf.reshape(tf.gather(self.b.beta, curr_time_step), [shape[0], 1, 1, 1, 1])
        sqa = tf.reshape(
            tf.gather(self.b.sqrt_alpha, curr_time_step), [shape[0], 1, 1, 1, 1]
        )
        ab = tf.reshape(
            tf.gather(self.b.alpha_bar, curr_time_step), [shape[0], 1, 1, 1, 1]
        )
        ab_prev = tf.reshape(
            tf.gather(self.b.alpha_bar_prev, curr_time_step), [shape[0], 1, 1, 1, 1]
        )
        sqab = tf.reshape(
            tf.gather(self.b.sqrt_alpha_bar, curr_time_step), [shape[0], 1, 1, 1, 1]
        )
        sqab_prev = tf.reshape(
            tf.gather(self.b.sqrt_alpha_bar_prev, curr_time_step),
            [shape[0], 1, 1, 1, 1],
        )
        sq1ab = tf.reshape(
            tf.gather(self.b.sqrt_one_minus_alpha_bar, curr_time_step),
            [shape[0], 1, 1, 1, 1],
        )
        x_0 = (x_t - sq1ab * pred_noise) / sqab
        # x_0 = pred_noise

        posterior_mean = (b * sqab_prev / (1 - ab)) * x_0 + (
            (1 - ab_prev) * sqa / (1 - ab)
        ) * x_t

        posterior_log_variance = (1 - ab_prev) * b / (1 - ab)

        return posterior_mean, posterior_log_variance

    def generate(self, shape=(1, 16, 16, 16, 16), last_step=0, context_value=None):
        # 0 create context samples
        context_input = tf.constant([[context_value]], dtype=tf.int64)

        # 1 random latent samples
        samples = tf.random.normal(shape=shape, dtype=tf.float32)


        # 2 iterate to denoise image latents
        for i in range(self.timesteps - 1, last_step - 1, -1):
            # 2a random noise at step i
            if i > 0:
                noise = tf.random.normal(shape=shape, dtype=tf.float32)
            else:
                noise = 0

            # 2b predict the noise over i steps
            curr_time_step = tf.repeat(i, shape[0])
            pred_noise = self.network([samples, curr_time_step, context_input])

            # 2c remove the predicted noise and add noise from above
            mean, var = self.sample(samples, pred_noise, curr_time_step, shape)
            mean = tf.clip_by_value(mean, -1, 1)
            samples = mean + tf.exp(0.5 * np.log(np.maximum(var, 1e-20))) * noise

        return samples

    def test(self, test_prefix, context=None):
        for i in [self.timesteps]:
            print(f"Generating for {i} rsteps")
            self.vqvae_trainer.load_weights(self.vqvae_load_ckpt)
            # To-Do: fix it
            if context is not None:
                img_latents = self.generate(
                    (10, 16, 16, 16, 64), last_step=self.timesteps - i, context_value=context
                )   
            else: 
                img_latents = self.generate(
                    (10, 16, 16, 16, 64), last_step=self.timesteps - i
                )
            images = self.vqvae_trainer.decoder(img_latents)
            np.save(
                f"./generated_images_dm3d/{test_prefix}-{i}rsteps.npy", images.numpy()
            )
        # return images
