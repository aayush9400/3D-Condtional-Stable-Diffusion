.. role:: raw-html(raw)
   :format: html

.. raw:: html

   <center><a href="https://summerofcode.withgoogle.com/programs/2022/projects/ZZQ6IrHq"><img src="https://developers.google.com/open-source/gsoc/resources/downloads/GSoC-logo-horizontal.svg" alt="gsoc" height="50"/></a></center>

.. raw:: html

   <center>
   <a href="https://summerofcode.withgoogle.com/projects/#6653942668197888"><img src="https://www.python.org/static/community_logos/python-logo.png" height="50"/></a>
   <a href="http://dipy.org"><img src="https://python-gsoc.org/logos/DIPY.png" alt="fury" height="50"/></a>
   </center>


Google Summer of Code Final Work Product
========================================

-  **Name:** Bayanagari Vara Lakshmi 
-  **Organization:** Python Software Foundation
-  **Sub-Organization:** DIPY
-  **Project:** `DIPY - Synthetic MRI generation <https://github.com/dipy/dipy/wiki/Google-Summer-of-Code-2023#project-4-creating-synthetic-mri-data>`_


Proposed Objectives
-------------------

* Human Brain MRI preprocessing function
* MRI reconstruction using `VQVAE <https://arxiv.org/pdf/1711.00937.pdf>`_
* Implement & train `Diffusion Model <https://arxiv.org/pdf/2006.11239.pdf>`_ on VQVAE latents
* Implement conditional Diffusion Model
* Generate conditional synthetic MRI
* Evaluate synthetic generations in `DIPY <https://github.com/dipy/dipy/wiki/Google-Summer-of-Code-2023#project-4-creating-synthetic-mri-data>`_


Modified Objectives(Additional)
-------------------------------

* 2D VQVAE on MNIST data
* 2D unconditional DDPM based LDM on MNIST data
* 3D VQVAE based on MONAI's PyTorch implementation
* 3D unconditional LDM based on MONAI's PyTorch implementation

Objectives Completed
--------------------

* Conducted Literature Review on limited existing diffusion modeling in Medical Imaging

  * Current literature [`1 <https://arxiv.org/pdf/2211.03364.pdf>`_, `2 <https://arxiv.org/pdf/2209.07162.pdf>`_] utilized VQGAN& DDPM models on MRNet, ADNI, Breast Cancer MRI, lung CT datasets
  * MONAI is the latest open-source platform with repositories on deep learning applications on BRATS & other medical imaging datasets, implemented in Pytorch
  * Our project serves as a source for easy, understandable & accessible implementation of anatomical MRI generation using unconditional Diffusion Modelling in Tensorflow

* Implemented `2D VQVAE <https://github.com/lb-97/GenerativeAI-VQVAE-MNIST#table-of-contents>`_ & `2D DDPM <https://github.com/lb-97/GenerativeAI-DDIM-MNIST/tree/main#readme>`_ based Latent Diffusion Model(LDM) on MNIST dataset & achieved perfect generations

* Worked on `CC359 <https://sites.google.com/view/calgary-campinas-dataset/home>`_ & `NFBS <http://preprocessed-connectomes-project.org/NFB_skullstripped/>`_ datasets, both consist of T1-weighted human brain MRI with 359 & 125 samples respectively. Preprocessed each input volume following the 3 steps below-

  * Skull-stripping the dataset, if required, using existing masks.
  * Pre-process using ``transform_img`` function - perform voxel resizing & affine transformation to obtain final (128,128,128,1) shape & (1,1,1) voxel shape
  * Neutralized background pixels to 0 using respective masks
  * MinMax normalization to rescale intensities to (0,1) 

* Implemented 3D versions of the above repositories from scratch

  * VQVAE3D

    * The encoder & decoder of 3D VQVAE are symmetrical with 3 Convolutional & 3 Transpose Convolutional layers respectively, followed by non-linear ``relu`` units
    * Vector Quantizer trains a learnable embedding matrix to identify closest latents for a given input based on L2 loss function
    * VQVAE gave superior results over VAE as shown in `this <https://arxiv.org/pdf/1711.00937.pdf>`_ paper, owing to the fact that quantizer addresses the problem of 'Posterior Collapse' seen in traditional VAEs
    * Trained the model for approximately 100 epochs using Adam optimizer with lr=1e-4, minimized reconstruction & quantizer losses together
    * Test dataset reconstructions-
    .. image:: https://github.com/dipy/dipy/blob/master/doc/_static/vqvae3d-reconst-f3.png
         :alt: VQVAE reconstructions on NFBS test dataset

  * 3D LDM

    * Built unconditional Latent Diffusion Model(LDM) combining `DDPM <https://arxiv.org/pdf/2006.11239.pdf>`_ & `Stable Diffusion <https://arxiv.org/pdf/2112.10752.pdf>`_ implementations
    * U-Net of the reverse process consists of 3 downsampling & 3 upsampling layers each consisting of 2 residual layers and an optional attention layer
    * Trained the model using linear (forward)variance scaling & various diffusion steps - 200, 300
    * Adopted `algorithm 4 <https://arxiv.org/pdf/2006.11239.pdf>`_ for sampling synthetic generations at 200 & 300 diffusion steps-
    .. image:: https://github.com/dipy/dipy/blob/master/doc/_static/dm3d-reconst-D200-D300.png
       :alt: 3D LDM synthetic generations
       :width: 800


* Adopted MONAI's implementation

  * Replaced VQVAE encoder & decoder with a slightly complex architecture that includes residual connections alternating between convolutions
  * Carried out experiments with same training parameters with varying batch sizes & also used both datasets in a single experiment


    .. image:: https://github.com/lb-97/dipy/blob/blog_branch_week_12_13/doc/_static/vqvae3d-monai-training-plots.png
       :alt: VQVAE-MONAI training plots
       :width: 800
     
  
  * Clearly the training curves show that the higher batch size & dataset length, the better the stability of the training metric for learning rate=1e-4
  * Plotted reconstructions for top two experiments - (Batch size=12, Both datasets) & (Batch size=5, NFBS dataset)


    .. image:: https://github.com/lb-97/dipy/blob/blog_branch_week_12_13/doc/_static/vqvae-reconstructions-comparison.png
       :alt: VQVAE-MONAI reconstructions on best performing models
       :width: 800
  
  * Existing diffusion model has been trained on these new latents to check for their efficacy on synthetic image generation
  * The training curves converged pretty quickly, but the sampled generations are still pure noise

    .. image:: https://github.com/lb-97/dipy/blob/blog_branch_week_12_13/doc/_static/dm3d-monai-training-curves.png
       :alt: 3D LDM training curve for various batch sizes & diffusion steps
       :width: 400
  * To summarize, we've stretched the capability of our VQVAE model despite being less complex with only ``num_res_channels=(32, 64)``. We consistently achieved improved reconstruction results with every experiment. Our latest experiments are trained using a weighted loss function with lesser weight attached to background pixels owing to their higher number. This led to not just capturing the outer structure of a human brain but also the volumetric details resembling microstructural information inside the brain. This is a major improvement from all previous trainings.

  * For future work we should look into two things - debugging Diffusion Model, scaling VQVAE model.

    * As a first priority, we could analyze the reason for pure noise output in DM3D generations, this would help us rule out any implementation errors of the sampling process.

    * As a second step, we could also try scaling up both VQVAE as well as the Diffusion Model in terms of complexity, such as increasing intermediate channel dimensions from 64 to 128 or 256. This hopefully may help us achieve the state-of-art on NFBS & CC359 datasets.


Objectives in Progress
----------------------

* Unconditional LDM hasn't shown any progress in generations yet. Increasing model complexity with larger number of intermediate channels & increasing diffusion steps to 1000 is a direction of improvement
* Implemented cross-attention module as part of U-Net, to accommodate conditional training such as tumor type, tumor location, brain age etc
* Implementation of evaluation metrics such as FID(Frechet Inception Distance) & IS(Inception Score) will be useful in estimating the generative capabilities of our models


Timeline
--------

.. list-table::
   :header-rows: 1

   * - Date
     - Description
     - Blog Post Link
   * - Week 0\  :raw-html:`<br>`\ (19-05-2023)
     - Journey of GSOC application & acceptance
     - `DIPY <https://github.com/dipy/dipy/blob/master/doc/posts/2023/2023_05_19_vara_week0.rst>`_
   * - Week 1\  :raw-html:`<br>`\ (29-05-2023)
     - Community bonding and Project kickstart
     - `DIPY <https://github.com/dipy/dipy/blob/master/doc/posts/2023/2023_05_29_vara_week1.rst>`_
   * - Week 2\  :raw-html:`<br>`\ (05-06-2023)
     - Deep Dive into VQVAE
     - `DIPY <https://github.com/dipy/dipy/blob/master/doc/posts/2023/2023_06_05_vara_week2.rst>`_
   * - Week 3\  :raw-html:`<br>`\ (12-06-2023)
     - VQVAE results and study on Diffusion models
     - `DIPY <https://github.com/dipy/dipy/blob/master/doc/posts/2023/2023_06_12_vara_week3.rst>`_
   * - Week 4\  :raw-html:`<br>`\ (19-06-2023)
     - Diffusion research continues
     - `DIPY <https://github.com/dipy/dipy/blob/master/doc/posts/2023/2023_06_19_vara_week4.rst>`_
   * - Week 5\  :raw-html:`<br>`\ (26-06-2023)
     - Carbonate HPC Account Setup, Experiment, Debug and Repeat
     - `DIPY <https://github.com/dipy/dipy/blob/master/doc/posts/2023/2023_06_26_vara_week5.rstt>`_
   * - Week 6 & Week 7\  :raw-html:`<br>`\ (10-07-2023)
     - Diffusion Model results on pre-trained VQVAE latents of NFBS MRI Dataset
     - `DIPY <https://github.com/dipy/dipy/blob/master/doc/posts/2023/2023_07_10_vara_week6_and_week7.rst>`_
   * - Week 8 & Week 9\  :raw-html:`<br>`\ (10-07-2023)
     - VQVAE MONAI models & checkerboard artifacts
     - `DIPY <>`_
   * - Week 10 & Week 11\  :raw-html:`<br>`\ (10-07-2023)
     - HPC issues, GPU availability, Tensorflow errors: Week 10 & Week 11
     - `DIPY <https://github.com/dipy/dipy/blob/master/doc/posts/2023/2023_08_07_vara_week_10_11.rst>`_
   * - Week 12 & Week 13\  :raw-html:`<br>`\ (10-07-2023)
     - Finalized experiments using both datasets
     - `DIPY <>`_
