import os
import glob

def get_dataset_list(*args, dataset_dir='/N/slate/aajais/skullstripping_datasets/'):
    dataset_list = []

    if args.dataset == 'CC':
        dataset_list = glob.glob(os.path.join(dataset_dir, 'CC359', 'Original', '*.nii.gz'))
    elif args.dataset == 'NFBS':
        dataset_list = glob.glob(os.path.join(dataset_dir, 'NFBS_Dataset', '*', 'sub-*_ses-NFB3_T1w_brain.nii.gz'))
    elif args.dataset == 'HCP':
        dataset_list = glob.glob(os.path.join(dataset_dir, 'HCP_T1', 'T1', '*.nii.gz'))
    elif args.dataset == 'both':
        dataset_list = glob.glob(os.path.join(dataset_dir, 'CC359', 'Original', '*.nii.gz'))
        dataset_list.extend(glob.glob(os.path.join(dataset_dir, 'NFBS_Dataset', '*', 'sub-*_ses-NFB3_T1w_brain.nii.gz')))
    elif args.dataset == 'all':
        dataset_list = glob.glob(os.path.join(dataset_dir, 'CC359', 'Original', '*.nii.gz'))
        dataset_list.extend(glob.glob(os.path.join(dataset_dir, 'NFBS_Dataset', '*', 'sub-*_ses-NFB3_T1w_brain.nii.gz')))
        dataset_list.extend(glob.glob(os.path.join(dataset_dir, 'HCP_T1', 'T1', '*.nii.gz')))

    return dataset_list
