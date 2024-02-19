import os
import random
import pickle
from tqdm import tqdm
import numpy as np
from dipy.io.image import load_nifti
from dipy.align.reslice import reslice
from fury.actor import slicer
from scipy.ndimage import affine_transform

import tensorflow as tf

def transform_img(image, affine, voxsize=None, init_shape=(256, 256, 256), scale=2):
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


def transform_img_brats(image, affine, voxsize, final_shape = (128, 128, 128)):
    temp_image, affine_temp = reslice(image, affine, voxsize, (2, 2, 2))
    temp_image = slicer(temp_image, affine_temp).resliced_array()

    current_shape = temp_image.shape

    pad_x = (final_shape[0] - current_shape[0]) // 2
    pad_y = (final_shape[1] - current_shape[1]) // 2
    pad_z = (final_shape[2] - current_shape[2]) // 2

    # Ensure the padding is equally distributed
    pad_width = ((pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z))

    transformed_img = np.pad(temp_image, pad_width, mode='constant', constant_values=0)

    return transformed_img, affine


def flip_axis_0(image, mask):
    """ Flip the image along axis 0 """
    factor = np.random.uniform(0.0, 1.0)
    if factor < 0.6:
        return image, mask
    else:
        return np.flip(image, 0), np.flip(mask, 0)


def adjust_brightness(image, factor_range=(0.8, 1.2)):
    """ Randomly adjust the brightness of the image within a given range """
    factor = np.random.uniform(factor_range[0], factor_range[1])
    image = np.clip(image * factor, 0, 1)  # Adjust brightness
    return image


def adjust_contrast(image, factor_range=(0.8, 1.2)):
    """ Randomly adjust the contrast of the image within a given range """
    factor = np.random.uniform(factor_range[0], factor_range[1])
    mean = np.mean(image)
    image = np.clip((1 + factor) * (image - mean) + mean, 0, 1)  # Adjust contrast
    return image


def load_transform_img(path):
    vol, affine, voxsize = load_nifti(path, return_voxsize=True)
    if 'CC359' in path:
        mask, _ = load_nifti(path.replace('Original', 'STAPLE').replace('.nii.gz', '_staple.nii.gz'))
        vol = vol*mask # zero out the background or non-region of interest areas.
    elif 'NFBS' in path:
        mask, _ = load_nifti(path[:-7]+'mask.nii.gz')
        vol = vol*mask # zero out the background or non-region of interest areas.
    elif 'BraTS2021' in path:
        vol = vol.astype(np.float32)
        mask, _ = load_nifti(path.replace('t1.nii.gz', 'seg.nii.gz'))
        mask = mask.astype(np.float32)
    else:
        # HCP Dataset
        mask, _ = load_nifti(path.replace('/T1/', '/T1_masks_evac/'))
        vol = vol*mask # zero out the background or non-region of interest areas.
        transform_vol = vol

    mask[mask < 1] = 0  # Values <1 in the mask is background

    if 'CC359' in path or 'NFBS' in path:
        if mask is not None:
            mask, _ = transform_img(mask, affine, voxsize)
        transform_vol, _ = transform_img(vol, affine, voxsize)
    elif 'BraTS2021' in path:
        if mask is not None:
            mask, _ = transform_img_brats(mask, affine, voxsize)
        transform_vol, _ = transform_img_brats(vol, affine, voxsize)
    
    mask[mask < 0] *= -1 # Handling negative pixels, occurred as a result of preprocessing
    mask = np.expand_dims(mask, -1)
    
    transform_vol[transform_vol < 0] *= -1 # Handling negative pixels, occurred as a result of preprocessing

    transform_vol = (transform_vol-np.min(transform_vol)) / \
        (np.max(transform_vol)-np.min(transform_vol))
    transform_vol = np.expand_dims(transform_vol, -1)

    return transform_vol, mask


def augmentDatasetHelperFunc(path):
    transform_vol, mask = None, None

    if isinstance(path, bytes):
        path = str(path.decode('utf-8'))

    transform_vol, mask = load_transform_img(path)

    augmented_transform_vol, augmented_mask = flip_axis_0(transform_vol, mask)
    augmented_transform_vol = adjust_brightness(augmented_transform_vol)  
    augmented_transform_vol = adjust_contrast(augmented_transform_vol)  

    return tf.convert_to_tensor(augmented_transform_vol, tf.float32), tf.convert_to_tensor(augmented_mask, tf.float32)


def datasetHelperFunc(path):
    transform_vol, mask = None, None

    if isinstance(path, bytes):
        path = str(path.decode('utf-8'))

    transform_vol, mask = load_transform_img(path)

    return tf.convert_to_tensor(transform_vol, tf.float32), tf.convert_to_tensor(mask, tf.float32)


def create_dataset(dataset_list, 
                   batch_size, 
                   dataset_save_path='/N/slate/aajais/skullstripping_datasets/training_data/', 
                   augment_flag=False,
                   save_flag=False):
    
    aug_size = int(len(dataset_list)*0.04)
    aug_path_list = random.sample(dataset_list, aug_size*batch_size)
    
    dataset = tf.data.Dataset.from_tensor_slices(dataset_list)
    # print(dataset.cardinality().numpy())
    dataset = (dataset
               .map(lambda x: tf.numpy_function(func=datasetHelperFunc, inp=[x], Tout=[tf.float32, tf.float32]), 
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
               .cache()
               .prefetch(tf.data.experimental.AUTOTUNE))
    
    if augment_flag:
        augmented_dataset = tf.data.Dataset.from_tensor_slices(aug_path_list)
        # print(augmented_dataset.cardinality().numpy())
        augmented_dataset = (augmented_dataset
                             .map(lambda x: tf.numpy_function(func=augmentDatasetHelperFunc, inp=[x], Tout=[tf.float32, tf.float32]), 
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
                             .cache()
                             .prefetch(tf.data.experimental.AUTOTUNE))
        
        print("Number of augmented images: ", augmented_dataset.cardinality().numpy())
        dataset = dataset.concatenate(augmented_dataset)
    
    print("Number of total images: ", dataset.cardinality().numpy())
    
    if save_flag and not os.path.exists(dataset_save_path):
        if tf.__version__ == '2.12.0':
            dataset.save(dataset_save_path)
        elif tf.__version__ == '2.9.1':
            tf.data.experimental.save(dataset, dataset_save_path, compression='GZIP')
            with open(dataset_save_path + '/element_spec', 'wb') as out_:  # also save the element_spec to disk for future loading
                pickle.dump(dataset.element_spec, out_)
        return 'Save Successful'
    else:
        print('returned tf.dataset')
        return dataset


def load_dataset(dataset_save_path):
    if tf.__version__ == '2.12.0':
        dataset = tf.data.Dataset.load(dataset_save_path)
        print('Dataset Loaded 2.12.0')
    elif tf.__version__ == '2.9.1':
        with open(dataset_save_path + '/element_spec', 'rb') as in_:
            espec = pickle.load(in_)

        dataset = tf.data.experimental.load(dataset_save_path, espec, compression='GZIP')
        print('Dataset Loaded 2.9.0')
    return dataset