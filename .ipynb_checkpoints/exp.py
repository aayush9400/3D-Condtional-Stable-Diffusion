import os
import glob
from tqdm import tqdm
import numpy as np
from dipy.io.image import load_nifti
from dipy.align.reslice import reslice
from scipy.ndimage import affine_transform

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
    print(transformed_img.shape, affine2.shape)
    return transformed_img, affine2


def datasetHelperFunc(path):
    transform_vol, mask = None, None
    if isinstance(path, bytes):
        path = str(path.decode('utf-8'))
    print(path)
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

    print(transform_vol.shape, mask.shape)
    return True


def get_dataset_list(dataset_dir='/N/slate/aajais/skullstripping_datasets/'):
    dataset_list = []
    dataset_list = glob.glob(os.path.join(dataset_dir, 'CC359', 'Original', '*.nii.gz'))
    with open('./dataset_NFBS.txt', 'r') as fd:
        for row in fd:
            dataset_list.append(dataset_dir + row.split('\n')[0])
    dataset_list.extend(glob.glob(os.path.join(dataset_dir, 'HCP_T1', 'T1', '*.nii.gz')))
    print(dataset_list)
    print('Total Images in dataset: ', len(dataset_list))
    return [dataset_list[0], dataset_list[400], dataset_list[-1]]


ds_list = get_dataset_list()
print(ds_list)
for path in ds_list:
    datasetHelperFunc(path)
