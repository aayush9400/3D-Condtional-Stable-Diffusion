import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
# import tensorflow_probability as tfp
import tensorflow as tf
from dipy.align.reslice import reslice


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
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.keras.initializers.HeUniform()
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
        if distribution:
            return distances
        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices


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
                 num_embeddings=32,
                 embedding_dim=64,
                 dropout=0.0,
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
        )

        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconst_loss")
        self.quantize_loss_tracker = keras.metrics.Mean(name="quantize_loss")

    def call(self, x):
        encoded_inputs = self.encoder(x)
        quantized = self.quantizer(encoded_inputs)
        decoder_outputs = self.decoder(quantized)
        return decoder_outputs

    def call_2(self, x):
        encoded_inputs = self.encoder(x)
        quantized = self.quantizer(encoded_inputs)
        tf.print(quantized.shape)
        return quantized

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.reconstruction_loss_tracker,
            self.quantize_loss_tracker
        ]

    def train_step(self, inputs):
        #print("printing: ", x, type(x), x.shape)
        x, mask = inputs
        reconstruction_loss = 0.0
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self(x)

            # Background pixels' value is multiplied by 0.5, their loss will be multiplied by 0.25
            if mask is not None:
                reconstructions = tf.where(
                    mask == 0, 0.5 * reconstructions, reconstructions)
                x = tf.where(mask == 0, 0.5 * x, x)
                # reconstructions[mask==0]*=0.25
                # x[mask==0]*=0.25

            # loss
            reconstruction_loss = tf.reduce_mean((reconstructions-x)**2)
            l = reconstruction_loss + self.quantizer.losses
            l = l/self.num_gpus

        grads = tape.gradient(l, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(l)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.quantize_loss_tracker.update_state(self.quantizer.losses)

        return {"loss": self.loss_tracker.result(),
                "reconst_loss": self.reconstruction_loss_tracker.result(),
                "quantize_loss": self.quantize_loss_tracker.result()}
    
    def test_step(self, data):
        x, mask = data

        # Outputs from the VQ-VAE
        reconstructions = self(x)

        # Apply mask if it exists
        if mask is not None:
            reconstructions = tf.where(mask == 0, 0.5 * reconstructions, reconstructions)
            x = tf.where(mask == 0, 0.5 * x, x)

        # Compute reconstruction loss
        reconstruction_loss = tf.reduce_mean((reconstructions - x)**2)
        loss = reconstruction_loss + self.quantizer.losses

        # Update metrics
        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.quantize_loss_tracker.update_state(self.quantizer.losses)

        return {"loss": self.loss_tracker.result(), 
                "reconst_loss": self.reconstruction_loss_tracker.result(), 
                "quantize_loss": self.quantize_loss_tracker.result()}
