import tensorflow as tf
from tensorflow.keras import layers, models


class VQEncoder(tf.keras.Model):
    """
    Encoder architecture that outputs a volume to a quantization layer. Also outputs skip connections
    """
    def __init__(self, num_channels, num_filters=8, embedding_dim=128, skip_connections=True, batchnorm=True):
        super(VQEncoder, self).__init__()

        self.skip = skip_connections

        if batchnorm:
            self.conv1 = tf.keras.Sequential([
                layers.Conv3D(num_filters, kernel_size=4, strides=2, padding='same', input_shape=(None, None, None, num_channels)),
                layers.BatchNormalization(),
                layers.ReLU()
            ])

            self.conv2 = tf.keras.Sequential([
                layers.Conv3D(num_filters * 2, kernel_size=4, strides=2, padding='same'),
                layers.BatchNormalization(),
                layers.ReLU()
            ])

            self.conv3 = tf.keras.Sequential([
                layers.Conv3D(num_filters * 4, kernel_size=4, strides=2, padding='same'),
                layers.BatchNormalization(),
                layers.ReLU()
            ])
        else:
            self.conv1 = tf.keras.Sequential([
                layers.Conv3D(num_filters, kernel_size=4, strides=2, padding='same', input_shape=(None, None, None, num_channels)),
                layers.ReLU()
            ])

            self.conv2 = tf.keras.Sequential([
                layers.Conv3D(num_filters * 2, kernel_size=4, strides=2, padding='same'),
                layers.ReLU()
            ])

            self.conv3 = tf.keras.Sequential([
                layers.Conv3D(num_filters * 4, kernel_size=4, strides=2, padding='same'),
                layers.ReLU()
            ])

        self.conv4 = layers.Conv3D(embedding_dim, kernel_size=4, strides=2, padding='same')

    def call(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        ze = self.conv4(x3)

        if self.skip:
            return x1, x2, x3, ze
        else:
            return ze


class VQDecoder_skip(tf.keras.Model):
    """
    Decoder architecture that accepts a volume from a quantization layer and skip connections from the encoder
    """
    def __init__(self, num_channels, num_filters=8, embedding_dim=32, batchnorm=False):
        super(VQDecoder_skip, self).__init__()

        if batchnorm:
            self.conv1 = tf.keras.Sequential([
                layers.Conv3DTranspose(num_filters * 4, kernel_size=4, strides=2, padding='same', input_shape=(None, None, None, embedding_dim)),
                layers.BatchNormalization(),
                layers.ReLU()
            ])

            self.conv2 = tf.keras.Sequential([
                layers.Conv3DTranspose(num_filters * 2, kernel_size=4, strides=2, padding='same'),
                layers.BatchNormalization(),
                layers.ReLU()
            ])

            self.conv3 = tf.keras.Sequential([
                layers.Conv3DTranspose(num_filters, kernel_size=4, strides=2, padding='same'),
                layers.BatchNormalization(),
                layers.ReLU()
            ])
        else:
            self.conv1 = tf.keras.Sequential([
                layers.Conv3DTranspose(num_filters * 4, kernel_size=4, strides=2, padding='same', input_shape=(None, None, None, embedding_dim)),
                layers.ReLU()
            ])

            self.conv2 = tf.keras.Sequential([
                layers.Conv3DTranspose(num_filters * 2, kernel_size=4, strides=2, padding='same'),
                layers.ReLU()
            ])

            self.conv3 = tf.keras.Sequential([
                layers.Conv3DTranspose(num_filters, kernel_size=4, strides=2, padding='same'),
                layers.ReLU()
            ])

        self.conv4 = layers.Conv3DTranspose(num_channels, kernel_size=4, strides=2, padding='same')

    def call(self, zq, encoder_layer1_output, encoder_layer2_output, encoder_layer3_output):
        x1 = self.conv1(zq)
        x2 = tf.concat([encoder_layer3_output, x1], axis=-1)
        x3 = self.conv2(x2)
        x4 = tf.concat([encoder_layer2_output, x3], axis=-1)
        x5 = self.conv3(x4)
        x6 = tf.concat([encoder_layer1_output, x5], axis=-1)
        x_recon = self.conv4(x6)

        return x_recon


class DoubleConv(tf.keras.Model):
    """
    Runs two convolutional layers, similar to the U-Net implementation
    """
    def __init__(self, filters_in, filters_out):
        super(DoubleConv, self).__init__()
        self.convnet = tf.keras.Sequential([
            layers.Conv3D(filters_out, kernel_size=3, padding='same', input_shape=(None, None, None, filters_in)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv3D(filters_out, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
    
    def call(self, x):
        return self.convnet(x)


class Down(tf.keras.Model):
    """
    Max pool then pass through two convolutional layers, used in UNet
    """
    def __init__(self, filters_in, filters_out):
        super(Down, self).__init__()
        self.convnet = tf.keras.Sequential([
            layers.MaxPool3D(pool_size=2, strides=2),
            DoubleConv(filters_in, filters_out)
        ])
    
    def call(self, x):
        return self.convnet(x)


class Up(tf.keras.Model):
    """
    Transpose convolution to upsample, then double convolution, used in UNet
    """
    def __init__(self, filters_in, filters_out, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = layers.UpSampling3D(size=2, interpolation='trilinear')
        else:
            self.up = layers.Conv3DTranspose(filters_out, kernel_size=2, strides=2)

        self.conv = DoubleConv(filters_in, filters_out)

    def call(self, x1, x2):
        x1 = self.up(x1)
        x = tf.concat([x2, x1], axis=-1)  # Concatenate in the channel dimension
        return self.conv(x)


class VectorQuantizerEMA(tf.keras.layers.Layer):
    def __init__(self, num_embeddings=512, embedding_dim=128, commitment_cost=6, decay=0.99, epsilon=1e-5, **kwargs):
        super(VectorQuantizerEMA, self).__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

    def build(self, input_shape):
        self.embeddings = self.add_weight(name='embeddings',
                                          shape=(self.embedding_dim, self.num_embeddings),
                                          initializer='uniform',
                                          trainable=True)
        self.ema_cluster_size = self.add_weight(name='ema_cluster_size',
                                                shape=(self.num_embeddings,),
                                                initializer='zeros',
                                                trainable=False)
        self.ema_w = self.add_weight(name='ema_w',
                                     shape=(self.embedding_dim, self.num_embeddings),
                                     initializer='uniform',
                                     trainable=False)

    def call(self, inputs, training=True):
        # inputs: (batch_size, height, width, depth, channels)
        inputs_flat = tf.reshape(inputs, [-1, self.embedding_dim])
        
        # Calculate distances between input and embeddings
        distances = (tf.reduce_sum(inputs_flat**2, axis=1, keepdims=True)
                     + tf.reduce_sum(self.embeddings**2, axis=0)
                     - 2 * tf.matmul(inputs_flat, self.embeddings))

        # Get closest embedding index
        encoding_indices = tf.argmin(distances, axis=1)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])
        
        # Quantize and calculate quantization loss
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, tf.shape(inputs))

        if training==True:
            # Update the embeddings
            updated_ema_cluster_size = self.ema_cluster_size * self.decay + \
                                    (1 - self.decay) * tf.reduce_sum(encodings, axis=0)

            dw = tf.matmul(encodings, inputs_flat, transpose_a=True)
            updated_ema_w = self.ema_w * self.decay + (1 - self.decay) * dw

            self.ema_cluster_size.assign(updated_ema_cluster_size)
            self.ema_w.assign(updated_ema_w)
            n = tf.reduce_sum(self.ema_cluster_size)
            self.embeddings.assign(self.ema_w / tf.reshape(self.ema_cluster_size, [1, -1]) * n / (n + self.epsilon))
        
        e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs)**2)
        loss = self.commitment_cost * e_latent_loss

        avg_probs = tf.reduce_mean(encodings, axis=0)
        perplexity = tf.exp(-tf.reduce_sum(avg_probs * tf.math.log(avg_probs + self.eps)))

        quantized = inputs + tf.stop_gradient(quantized - inputs)

        return loss, quantized, encoding_indices, perplexity

    def get_config(self):
        config = super(VectorQuantizerEMA, self).get_config()
        config.update({
            'num_embeddings': self.num_embeddings,
            'embedding_dim': self.embedding_dim,
            'commitment_cost': self.commitment_cost,
            'decay': self.decay,
            'epsilon': self.epsilon
        })
        return config
