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


def datasetHelperFunc(*args, path):
    transform_vol, mask = None, None
    if isinstance(path, bytes):
        path = str(path.decode('utf-8'))

    if 'CC359' in path:
        vol, affine, voxsize = load_nifti(path, return_voxsize=True)
        mask, _ = load_nifti(path.replace('Original', 'STAPLE').replace('.nii.gz', '_staple.nii.gz'))
        mask[mask < 1] = 0  # Values <1 in the mask is background
        vol = vol*mask # zero out the background or non-region of interest areas.
    elif 'NFBS' in path:
        vol, affine, voxsize = load_nifti(path, return_voxsize=True)
        mask, _ = load_nifti(path[:-7]+'mask.nii.gz')
        mask[mask < 1] = 0  # Values <1 in the mask is background
        vol = vol*mask # zero out the background or non-region of interest areas.
    else:
        vol, affine, voxsize = load_nifti(path, return_voxsize=True)
        mask, _ = load_nifti(path.replace('/T1/', '/T1_masks_evac/'))
        mask[mask < 1] = 0  # Values <1 in the mask is background
        vol = vol*mask # zero out the background or non-region of interest areas.
        if mask is not None:
            mask = np.expand_dims(mask, -1)
        transform_vol = (vol-np.min(vol)) / (np.max(vol)-np.min(vol))
        transform_vol = np.expand_dims(transform_vol, -1)

    if 'CC359' in path or 'NFBS' in path:
        if mask is not None:
            mask, _ = transform_img(mask, affine, voxsize)
            # Handling negative pixels, occurred as a result of preprocessing
            mask[mask < 0] *= -1
            mask = np.expand_dims(mask, -1)
        transform_vol, _ = transform_img(vol, affine, voxsize)
        # Handling negative pixels, occurred as a result of preprocessing
        transform_vol[transform_vol < 0] *= -1
        transform_vol = (transform_vol-np.min(transform_vol)) / \
            (np.max(transform_vol)-np.min(transform_vol))
        transform_vol = np.expand_dims(transform_vol, -1)

    if args.augment:
        augmented_transform_vol, augmented_mask = flip_axis_0(transform_vol, mask)
        augmented_transform_vol = adjust_brightness(augmented_transform_vol, 0.8)  # Increase brightness by 20%
        augmented_transform_vol = adjust_contrast(augmented_transform_vol, 0.8)  # Increase contrast
        return tf.convert_to_tensor(augmented_transform_vol, tf.float32), tf.convert_to_tensor(augmented_mask, tf.float32)
    else:
        return tf.convert_to_tensor(transform_vol, tf.float32), tf.convert_to_tensor(mask, tf.float32)
    
