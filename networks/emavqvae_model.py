import tensorflow as tf
from emavqvae import *

class VQVAE3D(tf.keras.Model):
    def __init__(self, num_channels, num_filters, embedding_dim=32, num_embeddings=512, skip_connections=True, batchnorm=False):
        super(VQVAE3D, self).__init__()
        self.skip = skip_connections
        self.encoder = VQEncoder(num_channels, num_filters, embedding_dim, skip_connections=skip_connections, batchnorm=batchnorm)
        self.quantization = VectorQuantizerEMA(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.decoder = VQDecoder_skip(num_channels, num_filters, embedding_dim, batchnorm)

    def call(self, x):
        x1, x2, x3, ze = self.encoder(x)
        loss, zq, embeddings, perplexity = self.quantization(ze)
        x_recon = self.decoder(zq, x1, x2, x3)

        return {'x_out': x_recon, 'vq_loss': loss, 'perplexity':perplexity}


class UNet(tf.keras.Model):
    def __init__(self, num_channels, num_filters=4, bilinear=True):
        super(UNet, self).__init__()
        # Encoding
        self.inconv = DoubleConv(num_channels, num_filters)
        self.down1 = Down(num_filters, num_filters * 2)
        self.down2 = Down(num_filters * 2, num_filters * 4)
        self.down3 = Down(num_filters * 4, num_filters * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(num_filters * 8, num_filters * 16 // factor)

        # Decoding
        self.up1 = Up(num_filters * 16, num_filters * 8 // factor, bilinear)
        self.up2 = Up(num_filters * 8, num_filters * 4 // factor, bilinear)
        self.up3 = Up(num_filters * 4, num_filters * 2 // factor, bilinear)
        self.up4 = Up(num_filters * 2, num_filters, bilinear)
        self.outconv = layers.Conv3D(num_channels, kernel_size=1, strides=1, padding='valid')

    def call(self, x):
        # Encoding
        xe1 = self.inconv(x)
        xe2 = self.down1(xe1)
        xe3 = self.down2(xe2)
        xe4 = self.down3(xe3)
        xe5 = self.down4(xe4)

        # Decoding
        xd4 = self.up1(xe5, xe4)
        xd3 = self.up2(xd4, xe3)
        xd2 = self.up3(xd3, xe2)
        xd1 = self.up4(xd2, xe1)
        out = self.outconv(xd1)

        return {'x_out': out}
