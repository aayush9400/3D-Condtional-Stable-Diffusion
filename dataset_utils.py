import os
import time
import glob
import random
import pickle
import numpy as np
import tensorflow as tf
from dipy.io.image import load_nifti
from dipy.align.reslice import reslice
from fury.actor import slicer
from scipy.ndimage import affine_transform


# Constants and Configuration
CONFIG = {
    'init_shape': (256, 256, 256),
    'final_shape_brats': (128, 128, 128),
    'scale': 2,
    'brightness_range': (0.9, 1.1),
    'contrast_range': (0.9, 1.1),
    'flip_chance': 0.6,
    'augment_percentage': 0.02,
    'dataset_save_path': "/N/slate/aajais/skullstripping_datasets/training_data/minus121_augment/"
}


def get_dataset_list(dataset_vers, test_run_flag, dataset_dir="/N/slate/aajais/skullstripping_datasets/"):
    dataset_list = []

    if dataset_vers == "CC":
        dataset_list = glob.glob(
            os.path.join(dataset_dir, "CC359", "Original", "*.nii.gz")
        )
    elif dataset_vers == "NFBS":
        dataset_list = glob.glob(
            os.path.join(
                dataset_dir, "NFBS_Dataset", "*", "sub-*_ses-NFB3_T1w_brain.nii.gz"
            )
        )
    elif dataset_vers == "HCP":
        dataset_list = glob.glob(os.path.join(dataset_dir, "HCP_T1", "T1", "*.nii.gz"))
    elif dataset_vers == "BraTS":
        dataset_list = glob.glob(
            os.path.join(dataset_dir, "BraTS2021", "*", "*_t1.nii.gz")
        )
    elif dataset_vers == "all":
        dataset_list = glob.glob(
            os.path.join(dataset_dir, "CC359", "Original", "*.nii.gz")
        )
        dataset_list.extend(
            glob.glob(
                os.path.join(
                    dataset_dir, "NFBS_Dataset", "*", "sub-*_ses-NFB3_T1w_brain.nii.gz"
                )
            )
        )
        dataset_list.extend(
            glob.glob(os.path.join(dataset_dir, "HCP_T1", "T1", "*.nii.gz"))
        )
    elif dataset_vers == "all-T":
        dataset_list = glob.glob(
            os.path.join(dataset_dir, "CC359", "Original", "*.nii.gz")
        )
        dataset_list.extend(
            glob.glob(
                os.path.join(
                    dataset_dir, "NFBS_Dataset", "*", "sub-*_ses-NFB3_T1w_brain.nii.gz"
                )
            )
        )
        dataset_list.extend(
            glob.glob(os.path.join(dataset_dir, "HCP_T1", "T1", "*.nii.gz"))
        )
        dataset_list.extend(
            glob.glob(os.path.join(dataset_dir, "BraTS2021", "*", "*_t1.nii.gz"))
        )

    if test_run_flag:
        print(len(dataset_list))
        dataset_list = dataset_list[:24]

    return dataset_list


def transform_image(image, affine, voxsize=None, scale=CONFIG['scale']):
    if voxsize is not None:
        image2, affine2 = reslice(image, affine, voxsize, (1, 1, 1))
    initial_shape = CONFIG['init_shape']
    affine2[:3, 3] += np.array(
        [initial_shape[0] // 2, initial_shape[1] // 2, initial_shape[2] // 2]
    )
    inv_affine = np.linalg.inv(affine2)
    transformed_img = affine_transform(image2, inv_affine, output_shape=initial_shape)
    transformed_img, _ = reslice(
        transformed_img, np.eye(4), (1, 1, 1), (scale, scale, scale)
    )
    return transformed_img, affine2


def transform_brats_image(image, affine, voxsize, final_shape=CONFIG["final_shape_brats"]):
    temp_image, affine_temp = reslice(image, affine, voxsize, (2, 2, 2))
    temp_image = slicer(temp_image, affine_temp).resliced_array()

    current_shape = temp_image.shape

    pad_x = (final_shape[0] - current_shape[0]) // 2
    pad_y = (final_shape[1] - current_shape[1]) // 2
    pad_z = (final_shape[2] - current_shape[2]) // 2

    pad_width = ((pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z))
    transformed_img = np.pad(temp_image, pad_width, mode="constant", constant_values=0)

    return transformed_img, affine


def flip_axis_0(image, mask):
    """Flip the image along axis 0"""
    if random.random() < CONFIG['flip_chance']:
        return image, mask
    else:
        return np.flip(image, 0), np.flip(mask, 0)


def adjust_brightness(image):
    """
    Adjust the brightness of an image.
    """
    factor = np.random.uniform(*CONFIG['brightness_range'])
    return np.clip(image * factor, 0, 1)


def adjust_contrast(image):
    """
    Adjust the contrast of an image.
    """
    factor = np.random.uniform(*CONFIG['contrast_range'])
    mean = np.mean(image)
    return np.clip((1 + factor) * (image - mean) + mean, 0, 1)


def load_transform_img(path):
    vol, affine, voxsize = load_nifti(path, return_voxsize=True)
    context = np.zeros(1)
    if "CC359" in path:
        mask, _ = load_nifti(
            path.replace("Original", "STAPLE").replace(".nii.gz", "_staple.nii.gz")
        )
        vol = vol*mask # zero out the background or non-region of interest areas.
    elif "NFBS" in path:
        mask, _ = load_nifti(path[:-7] + "mask.nii.gz")
        vol = vol*mask # zero out the background or non-region of interest areas.
    elif "BraTS2021" in path:
        vol = vol.astype(np.float32)
        mask, _ = load_nifti(path.replace("t1.nii.gz", "seg.nii.gz"))
        # new_mask = mask.copy()
        # new_mask[vol > 0] = 1
        # new_mask[vol == 0] = 0
        # new_mask[mask >= 1] = 1.5
        mask = mask.astype(np.float32)
        context = np.ones(1)
    else:
        # HCP Dataset
        transform_vol = vol
        mask = np.zeros_like(transform_vol)

    if not "BraTS2021" in path:
        transform_vol, _ = transform_image(vol, affine, voxsize)
        mask = np.zeros_like(transform_vol)
    elif "BraTS2021" in path:
        if mask is not None:
            mask, _ = transform_brats_image(mask, affine, voxsize)
            mask[mask < 0] *= -1 
            mask[mask >= 1] = 1
        transform_vol, _ = transform_brats_image(vol, affine, voxsize)

    mask = np.expand_dims(mask, -1)

    transform_vol[
        transform_vol < 0
    ] *= -1  # Handling negative pixels, occurred as a result of preprocessing

    transform_vol = (transform_vol - np.min(transform_vol)) / (
        np.max(transform_vol) - np.min(transform_vol)
    )
    # transform_vol = 2 * transform_vol - 1
    transform_vol = np.expand_dims(transform_vol, -1)
    context = np.expand_dims(context, -1)
    return transform_vol, mask, context


def augmentDatasetHelperFunc(path):
    transform_vol, mask, context = None, None, None

    if isinstance(path, bytes):
        path = str(path.decode("utf-8"))

    transform_vol, mask, context = load_transform_img(path)

    augmented_transform_vol, augmented_mask = flip_axis_0(transform_vol, mask)
    augmented_transform_vol = adjust_brightness(augmented_transform_vol)
    augmented_transform_vol = adjust_contrast(augmented_transform_vol)

    return tf.convert_to_tensor(
        transform_vol, tf.float32
    ), tf.convert_to_tensor(mask, tf.float32), tf.convert_to_tensor(context, tf.int64)


def datasetHelperFunc(path):
    transform_vol, mask, context = None, None, None

    if isinstance(path, bytes):
        path = str(path.decode("utf-8"))

    transform_vol, mask, context = load_transform_img(path)

    return tf.convert_to_tensor(
        transform_vol, tf.float32
    ), tf.convert_to_tensor(mask, tf.float32), tf.convert_to_tensor(context, tf.int64)


def create_dataset(
    dataset_list,
    batch_size,
    dataset_save_path=CONFIG['dataset_save_path'],
    augment_flag=False,
    save_flag=False,
):
    print('Running Dataset Operations')
    dataset_list += dataset_list[-298:]
    dataset = tf.data.Dataset.from_tensor_slices(dataset_list)
    dataset = (
        dataset.map(
            lambda x: tf.numpy_function(
                func=datasetHelperFunc, inp=[x], Tout=[tf.float32, tf.float32, tf.int64]
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    if augment_flag:
        aug_size = int(len(dataset_list) * 0.04)
        aug_path_list = random.sample(dataset_list, aug_size * batch_size)
        augmented_dataset = tf.data.Dataset.from_tensor_slices(aug_path_list)
        augmented_dataset = (
            augmented_dataset.map(
                lambda x: tf.numpy_function(
                    func=augmentDatasetHelperFunc,
                    inp=[x],
                    Tout=[tf.float32, tf.float32, tf.int64],
                ),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        print("Number of augmented images: ", augmented_dataset.cardinality().numpy())
        dataset = dataset.concatenate(augmented_dataset)

    print("Number of total images: ", dataset.cardinality().numpy())

    if save_flag:
        dataset.save(dataset_save_path)
        return "Save Successful"
    else:
        print("Returned tf.dataset.Dataset")
        return dataset


def load_dataset(dataset_save_path):
    if tf.__version__ == "2.12.0":
        dataset = tf.data.Dataset.load(dataset_save_path)
        print("Dataset Loaded 2.12.0")
    elif tf.__version__ == "2.9.1":
        with open(dataset_save_path + "/element_spec", "rb") as in_:
            espec = pickle.load(in_)

        dataset = tf.data.experimental.load(
            dataset_save_path, espec, compression="GZIP"
        )
        print("Dataset Loaded 2.9.0")
    return dataset


if __name__=="__main__":
    dataset_list = get_dataset_list(dataset_vers="all-T", test_run_flag=False)
    print("Total Images in dataset: ", len(dataset_list))

    start = time.time()
    create_flag = create_dataset(
        dataset_list=dataset_list,
        batch_size=12,
        dataset_save_path=CONFIG['dataset_save_path'],
        augment_flag=True,
        save_flag=True,
    )
    end = time.time()
    print(create_flag, "time taken:", (end - start) / 60)