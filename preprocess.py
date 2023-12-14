import numpy as np
import tensorflow as tf

from dipy.io.image import load_nifti
from dipy.align.reslice import reslice
from scipy.ndimage import affine_transform


def transform_img(image, affine, voxsize=None, init_shape=(256, 256, 256), scale=2):
    """ Transform and Reslice Image """
    if voxsize is not None:
        image2, affine2 = reslice(image, affine, voxsize, (1, 1, 1))

    affine2[:3, 3] += np.array([init_shape[0]//2,
                                init_shape[1]//2,
                                init_shape[2]//2])
    inv_affine = np.linalg.inv(affine2)
    transformed_img = affine_transform(
        image2, inv_affine, output_shape=init_shape)
    transformed_img, _ = reslice(transformed_img, np.eye(4), (1, 1, 1),
                                 (scale, scale, scale))
    return transformed_img, affine2


def flip_axis_0(image, mask):
    """ Flip the image along axis 0 """
    return np.flip(image, 0), np.flip(mask, 0)


def adjust_brightness(image, factor):
    """ Adjust the brightness of the image """
    image = np.clip(image, 0, 1)
    image = np.clip(image * factor, 0, 1)  # Adjust brightness
    return image


def adjust_contrast(image, factor):
    """ Adjust the contrast of the image """
    image = np.clip(image, 0, 1)
    mean = np.mean(image)
    image = np.clip((1 + factor) * (image - mean) + mean, 0, 1)  # Adjust contrast
    return image


def load_img(path, return_voxsize=False):
    return load_nifti(path, return_voxsize)