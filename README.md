# SyntheticMRI

Synthetic MRI generation using StableDiffusion &amp; VQVAE

## Proposal
The aim of this project is to contribute to the open source community in synthetic image generation for MRI images, using latent diffusion modelling inspired by [Denoised Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf) &amp; [Stable Diffusion](https://arxiv.org/abs/2112.10752). 

The advantages of Diffusion Models have been witnessed only recently in the field of Medical imaging, leaving scope for more exploration. We aim to produce large size synthetic datasets that could be used to pre-train data hungry Transformer models for downstream tasks. This could help avoid training models on small medical datasets collected from different machines with varying parameters, nor rely on Imagenet pre-trained models that suffer from change in domain distributional shift.

## Introduction
In the context of deep learning, Diffusion Models consist of two processes - forward & reverse. These two processes are designed as Markov chain models where future state only depends on the current state & not on past states. Each state is modelled as a Gaussian distribution with mean & variance as the model parameters. Forward processes denote gradual addition of noise to an input image, where as reverse process denotes gradual removal of noise from a random noisy image.

![Diffusion Processes](https://github.com/lb-97/SyntheticMRI/blob/main/_static/diffusionmodel.png)

In the forward process, these model parameters are not learnt rather chosen as hyperparameters. So, reparametrization trick can be used to calculate forward probabilities at every timestep. Whereas, the paper proposes a U-Net to learn the parameters of the reverse process. Since the ouptut of the final state in the reverse process is a probability function, the objective therefore is to increase the likelihood of the probability function. Using KL divergence, the paper proves that maximizing likelihood function is equivalent to decreasing the distance between the forward posterior gaussian & the reverse gaussian at each timestep. On further derivation, it also shows that the objective function is eventually equivalent to predicting the noise added at each step. Therefore U-Net is trained to predict the gaussian noise added at each timestep using L2 loss.

```
Maximizing likelihood function = Minimizing L2 loss of noise prediction.
That's why the whole training methodology is also called denoised score matching!
```

This is taken a notch up by carrying out training in latent space, inspired by Stable Diffusion. This paper proves that seperating training of perceptual compression(downsampling original dimension to latent space dimension) & denoising mechanism results in stable training, hence the name 'Stable Diffusion'. Perceptual compression is achieved through encoder-decoder models such as KL-VAE, VQ-VAE. VQ-VAE model addresses ['Posterior collapse'](https://github.com/lb-97/GenerativeAI-VQVAE-MNIST/blob/main/README.md) observed in traditional VAEs by effectively utilizing the latent space. For our problem statement, we pre-train VQ-VAE on same medical datasets until training curve is stabilized & converged. 

## Models
As the first step in training, we designed a 3D VQVAE that consists of an encoder, decoder, and a quantizer. Encoder downsamples the input thrice using strided convolutions each followed by ``relu`` non-linear units. So an input size of (128,128,128,1) is reduced to (16,16,16,16) with latent_dim=16, the number of channels in the latents. The quantizer is initialized to a learnable embedding matrix of size 16*128 i.e., 128 embeddings each of size latent_dim=16. It is trained to quantize encoder outputs to the closest of these embeddings using L2 loss function. The decoder consists of upsampling layers using transposed convolutions each followed by ``relu`` non-linear units. 

The model is trained to minimize the sum of reconstruction loss & quantization loss. The architecture of the model is inspired by [tensorflow's official implementation of VQ-VAE for 2D images](https://keras.io/examples/generative/vq_vae/). This model has been tested on MNIST dataset and the reconstruction results can be seen [here](https://github.com/lb-97/GenerativeAI-VQVAE-MNIST/blob/main/README.md#Results).

The basic code blocks of the U-Net of the Diffusion Model are Downsampling, Middle and Upsampling blocks, where each constitute ResidualBlock & AttentionBlock. ResidualBlock is additionally conditioned on the diffusion timestep, DDPM implements this conditioning by adding diffusion timestep to the input image, whereas DDIM performs a concatenation.

Downsampling & Upsampling in the U-Net are performed 4 times with decreasing & increasing widths respectively. Each downsampling layer consists of two ResidualBlocks, an optional AttentionBlock and a convolutional downsampling(stride=2) layer. At each upsampling layer, there's a concatenation from the respective downsampling layer, three ResidualBlocks, an optional AttentionBlock, ``keras.layers.Upsampling2D`` and a Conv2D layers. The Middle block consists of two ResidualBlocks with an AttentionBlock in between, resulting in no change in the output size. The final output of the Upsampling block is followed by a GroupNormalization layer, Swish Activation layer and Conv2D layer to provide an output with desired dimensions.

The U-Net model is trained to minimize the noise predicted at every timestep for every image. We produce latents from VQVAE and train unconditional U-Net in latent dimension to produce better generations than that of VQVAE. This model has been tested on MNIST dataset and the generations obtained can be seen [here](https://github.com/dipy/dipy/blob/master/doc/_static/DM-MNIST-DDIM300-108epoch.png).

## Experiments
We ran multiple 3D VQVAE experiments with varying hyper-parameters (downsampling factor f, batch_size B) on [NFBS dataset](http://preprocessed-connectomes-project.org/NFB_skullstripped/). The best results achieved are for [f=3 & B=10](https://github.com/dipy/dipy/blob/master/doc/_static/vqvae-f3-higher-epochs.png). The diffusion model results on these trained latents were noisy images with no insights into the generations.

To improve the efficacy of VQVAE latents further, we adopted MONAI's encoder-decoder architecture, that has residual connection after every convolutional layer. This model increases the complexity through skip connections facilitating the flow of non-zero gradients in backpropagation. This resulted in high qualitative & quantitative reconstructions.

A summary of all the experiements & outputs have been highlighted in [this GitHub Gist](https://gist.github.com/lb-97/57347e7d06d87a0aa3b77887631f33bc)










