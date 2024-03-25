import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf


class VectorQuantizer(layers.Layer):
    """
        VectorQuantizer is used to quantize the latents from VQVAE encoder
    """

    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

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
        return quantized

    def get_code_indices(self, flattened_inputs, distribution=False):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Can retrieve all distances to analyze the distribution of the ori input.
        if distribution:
            return distances

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

# Used additional residual connections post the convolution layers to increase the
# complexity of the model


def get_encoder_2(latent_dim=16, shape=(128, 128, 128, 1)):
    encoder_inputs = keras.Input(shape=shape)
    x = layers.Conv3D(32, 4, activation='relu', strides=2,
                      padding='same')(encoder_inputs)
    x = layers.Conv3D(64, 4, activation='relu', strides=2, padding='same')(x)

    y = layers.Conv3D(128, 3, activation='relu', padding='same')(x)
    x = layers.Conv3D(128, 1)(x) + y

    y = layers.Conv3D(128, 3, activation='relu', padding='same')(x)
    x = layers.Conv3D(128, 1)(x) + y

    encoder_outputs = layers.Conv3D(latent_dim, 1, padding='same')(x)
    return keras.Model(encoder_inputs, encoder_outputs, name='encoder_2')


def get_encoder(latent_dim=16, shape=(128, 128, 128, 1), down=3):
    encoder_inputs = keras.Input(shape=shape)
    x = layers.Conv3D(32, 3, activation="relu", strides=2, padding="same")(
        encoder_inputs
    )
    x = layers.Conv3D(64, 3, activation="relu", strides=2, padding="same")(x)
    if down == 3:
        x = layers.Conv3D(128, 3, activation="relu",
                          strides=2, padding="same")(x)
    encoder_outputs = layers.Conv3D(
        latent_dim, 1, padding="same")(x)  # (16,16,16,latent_dim)
    return keras.Model(encoder_inputs, encoder_outputs, name="encoder")


# Used additional residual connections prior to the convolution layers to increase the
# complexity of the model, just as in encoder_2
def get_decoder_2(latent_dim=16, shape=(128, 128, 128, 1)):
    latent_inputs = keras.Input(shape=get_encoder_2(
        latent_dim, shape).output.shape[1:])

    x = layers.Conv3DTranspose(128, 1, activation='relu')(latent_inputs)

    y = layers.Conv3DTranspose(128, 3, activation='relu', padding='same')(x)
    x = layers.Conv3DTranspose(128, 1)(y) + x

    y = layers.Conv3DTranspose(128, 3, activation='relu', padding='same')(x)
    x = layers.Conv3DTranspose(128, 1)(y) + x

    x = layers.Conv3DTranspose(
        64, 4, activation='relu', strides=2, padding='same')(x)

    x = layers.Conv3DTranspose(
        32, 4, activation='relu', strides=2, padding='same')(x)

    decoder_outputs = layers.Conv3DTranspose(shape[-1], 3, padding="same")(x)
    return keras.Model(latent_inputs, decoder_outputs, name='decoder_2')


def get_decoder(latent_dim=16, shape=(128, 128, 128, 1), down=3):
    latent_inputs = keras.Input(shape=get_encoder(
        latent_dim, shape=shape).output.shape[1:])
    if down == 3:
        x = layers.Conv3DTranspose(128, 3, activation="relu", strides=2, padding="same")(
            latent_inputs
        )
    else:
        x = latent_inputs
    x = layers.Conv3DTranspose(
        64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv3DTranspose(
        32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv3DTranspose(shape[-1], 3, padding="same")(x)
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")


def get_vqvae(latent_dim=16, num_embeddings=64, shape=(28, 28, 1), down=3):
    vq_layer = VectorQuantizer(
        num_embeddings, latent_dim, name="vector_quantizer")
    encoder = get_encoder(latent_dim, shape=shape, down=down)
    decoder = get_decoder(latent_dim, shape=shape, down=down)
    inputs = keras.Input(shape=shape)
    encoder_outputs = encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs)
    reconstructions = decoder(quantized_latents)
    return keras.Model(inputs, reconstructions, name="vq_vae")


def get_vqvae_2(latent_dim=16, num_embeddings=64, shape=(128, 128, 128, 1)):
    vq_layer = VectorQuantizer(
        num_embeddings, latent_dim, name="vector_quantizer")
    encoder = get_encoder_2(latent_dim, shape=shape)
    decoder = get_decoder_2(latent_dim, shape=shape)
    inputs = keras.Input(shape=shape)
    encoder_outputs = encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs)
    reconstructions = decoder(quantized_latents)
    return keras.Model(inputs, reconstructions, name="vq_vae_2")


class VQVAETrainer(keras.models.Model):
    def __init__(self, train_variance=0.0949, latent_dim=32, num_embeddings=128, shape=(128, 128, 128, 1), args=None, **kwargs):
        super().__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.args = args

        if args.vqvae_mode == 1:
            self.vqvae = get_vqvae(
                self.latent_dim, self.num_embeddings, shape, down=args.down)
        else:
            self.vqvae = get_vqvae_2(
                self.latent_dim, self.num_embeddings, shape)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")
        # print(self.vqvae.summary())

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            reconstruction_loss = (
                # / self.train_variance
                tf.reduce_mean((x - reconstructions) ** 2)
            )
            total_loss = reconstruction_loss + sum(self.vqvae.losses)
            total_loss = total_loss/self.args.num_gpus  # new line

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }

    # Testing
    def test_step(self, x):
        print(x.shape)
        print(tf.shape(x), type(x))
        reconst = self.vqvae(x)
        np.save(
            f'./reconst_vqvae3d/reconst3d-{self.args.suffix}.npy', reconst.numpy())
        loss = tf.reduce_mean((x - reconst) ** 2) / self.train_variance
        return {"loss": loss}
