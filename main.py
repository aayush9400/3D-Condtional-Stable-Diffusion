import argparse
import os
import gc
import glob
import numpy as np
import random
from tqdm import tqdm

from tensorflow import keras
from tensorflow.keras import layers
# import tensorflow_probability as tfp
import tensorflow as tf

from dipy.io.image import load_nifti
from dipy.align.reslice import reslice
from fury.actor import slicer
from scipy.ndimage import affine_transform

from vqvae3d_monai import VQVAE
from dm3d import DiffusionModel

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1 = INFO, 2 = WARNING, 3 = ERROR)
# tf.get_logger().setLevel('ERROR')


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


def datasetHelperFunc(path):
    transform_vol, mask = None, None
    if isinstance(path, bytes):
        path = str(path.decode('utf-8'))
    vol, affine, voxsize = load_nifti(path, return_voxsize=True)
    if 'CC359' in path:
        mask, _ = load_nifti(path.replace('Original', 'STAPLE').replace('.nii.gz', '_staple.nii.gz'))
    elif 'NFBS' in path:
        mask, _ = load_nifti(path[:-7]+'mask.nii.gz')
    elif 'BraTS2021' in path:
        vol = vol.astype(np.float32)
        mask, _ = load_nifti(path.replace('t1.nii.gz', 'seg.nii.gz'))
        mask = mask.astype(np.float32)
    else:
        # HCP Dataset
        mask, _ = load_nifti(path.replace('/T1/', '/T1_masks_evac/'))

    mask[mask < 1] = 0  # Values <1 in the mask is background
    vol = vol*mask # zero out the background or non-region of interest areas.

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

    if args.augment:
        # print('augmented dataset!')
        augmented_transform_vol, augmented_mask = flip_axis_0(transform_vol, mask)
        augmented_transform_vol = adjust_brightness(augmented_transform_vol)  
        augmented_transform_vol = adjust_contrast(augmented_transform_vol)  
        return tf.convert_to_tensor(augmented_transform_vol, tf.float32), tf.convert_to_tensor(augmented_mask, tf.float32)
    else:
        return tf.convert_to_tensor(transform_vol, tf.float32), tf.convert_to_tensor(mask, tf.float32)


def get_dataset_list(dataset_dir='/N/slate/aajais/skullstripping_datasets/'):
    dataset_list = []

    if args.dataset == 'CC':
        dataset_list = glob.glob(os.path.join(dataset_dir, 'CC359', 'Original', '*.nii.gz'))
    elif args.dataset == 'NFBS':
        dataset_list = glob.glob(os.path.join(dataset_dir, 'NFBS_Dataset', '*', 'sub-*_ses-NFB3_T1w_brain.nii.gz'))
    elif args.dataset == 'HCP':
        dataset_list = glob.glob(os.path.join(dataset_dir, 'HCP_T1', 'T1', '*.nii.gz'))
    elif args.dataset == 'BraTS':
        dataset_list = glob.glob(os.path.join(dataset_dir, 'BraTS2021', '*', '*_t1.nii.gz'))
    elif args.dataset == 'all':
        dataset_list = glob.glob(os.path.join(dataset_dir, 'CC359', 'Original', '*.nii.gz'))
        dataset_list = glob.glob(os.path.join(dataset_dir, 'NFBS_Dataset', '*', 'sub-*_ses-NFB3_T1w_brain.nii.gz'))
        dataset_list = glob.glob(os.path.join(dataset_dir, 'HCP_T1', 'T1', '*.nii.gz'))
    elif args.dataset == 'all-T':
        dataset_list = glob.glob(os.path.join(dataset_dir, 'CC359', 'Original', '*.nii.gz'))
        dataset_list.extend(glob.glob(os.path.join(dataset_dir, 'NFBS_Dataset', '*', 'sub-*_ses-NFB3_T1w_brain.nii.gz')))
        dataset_list.extend(glob.glob(os.path.join(dataset_dir, 'HCP_T1', 'T1', '*.nii.gz')))
        dataset_list = glob.glob(os.path.join(dataset_dir, 'BraTS2021', '*', '*_t1.nii.gz'))

    if args.test_run:
        dataset_list = dataset_list[:24]
    return dataset_list


def run(args):
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    args.num_gpus = strategy.num_replicas_in_sync
    print(f'Number of devices: {args.num_gpus}')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    print(tf.config.list_logical_devices())
        
    args.bs = args.lbs*args.num_gpus
    args.suffix = 'B'+str(args.bs)

    if args.kernel_resize:
        args.suffix += '-KR'
    if args.augment:
        args.suffix += '-AUG'
    if args.dataset=='CC':
        args.suffix = 'CC-'+args.suffix
    elif args.dataset=='NFBS':
        args.suffix = 'NFBS-'+args.suffix
    elif args.dataset=='HCP':
        args.suffix = 'HCP-'+args.suffix
    elif args.dataset=='BraTS':
        args.suffix = 'BraTS-'+args.suffix
    elif args.dataset == 'all':
        args.suffix += '-all'
    elif args.dataset == 'all-T':
        args.suffix += '-all-T'
    print('Global Batch Size: ', args.bs)

    dataset_list = get_dataset_list()
    print('Total Images in dataset: ', len(dataset_list))
    args.test_size = len(dataset_list) - (len(dataset_list)//args.bs)*args.bs
    print(args)

    lis = dataset_list[:-args.test_size] if args.test_size > 0 else dataset_list[:]
    if args.test_vq or args.test_dm:
        lis = dataset_list[-args.test_size:]

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    dataset = tf.data.Dataset.from_tensor_slices(lis)
    
    dataset = dataset.map(lambda x: tf.numpy_function(func=datasetHelperFunc, inp=[x], Tout=[tf.float32, tf.float32]),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    if args.augment and args.train_vq:
        augmented_dataset = tf.data.Dataset.from_tensor_slices(random.sample(lis, int(len(lis)*0.04)*args.bs))
        augmented_dataset = augmented_dataset.map(
            lambda x: tf.numpy_function(func=datasetHelperFunc, inp=[x], Tout=[tf.float32, tf.float32]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        print("Number of augmented images: ", augmented_dataset.cardinality().numpy())
        dataset = dataset.concatenate(augmented_dataset)
        print("Number of total images: ", dataset.cardinality().numpy())
    # dataset = dataset.cache()
    if args.train_vq:
        train_size = int((1 - args.val_perc) * dataset.cardinality().numpy())
        train_size = train_size - (train_size % args.bs)

        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size)

        train_dataset = train_dataset.batch(args.bs).prefetch(tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.batch(args.bs).prefetch(tf.data.experimental.AUTOTUNE)

        train_dataset = train_dataset.with_options(options)
        val_dataset = val_dataset.with_options(options)

        train_dataset_cardinality = train_dataset.cardinality().numpy()
        val_dataset_cardinality = val_dataset.cardinality().numpy()

        print(f"Number of images in the training dataset: {train_dataset_cardinality * args.bs}")
        print(f"Number of images in the validation dataset: {val_dataset_cardinality * args.bs}")

        print(f'Training Scaled VQVAE monai')
        with strategy.scope():
            model = VQVAE(
                in_channels=1,
                out_channels=1,
                num_channels=(32, 64, 128),
                num_res_channels=(32, 64, 128),
                num_res_layers=3,
                # downsample_parameters=(stride, kernel_size, dilation_rate, padding)
                downsample_parameters=((2, 4, 1, 'same'), (2, 4, 1, 'same'), (2, 4, 1, 'same')),
                upsample_parameters=((2, 4, 1, 'same', 0), (2, 4, 1, 'same', 0), (2, 4, 1, 'same', 0)),
                num_embeddings=256,
                embedding_dim=64,
                num_gpus=args.num_gpus,
                kernel_resize=args.kernel_resize)

            x = tf.keras.layers.Input(shape=(128, 128, 128, 1))
            m = tf.keras.Model(inputs=[x], outputs=model(x))
            print(m.summary())

            # Compile the model
            # Loss is implemented in model file
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=args.lr),
            )

            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=f'/N/slate/aajais/checkpoints-vqvae-monai-scaled-128/{args.suffix}/'+'{epoch}.ckpt',
                save_weights_only=True,
                save_best_only=args.save_best_only)
            
            csv_logger = tf.keras.callbacks.CSVLogger(f'/N/slate/aajais/checkpoints-vqvae-monai-scaled-128/{args.suffix}/training.log', append=True)

            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='reconst_loss', 
                              factor=0.2, 
                              patience=5, 
                              min_delta=1e-5, 
                              cooldown=2, 
                              min_lr=1e-6, 
                              verbose=1)
            
            if args.test_run:
                callbacks = [reduce_lr]
            else:
                callbacks = [model_checkpoint_callback, csv_logger, reduce_lr]


        initial_epoch = 0
        if args.resume_ckpt:
            model.load_weights(
                f'/N/slate/aajais/checkpoints-vqvae-monai-scaled-128/{args.suffix}/'+args.resume_ckpt+'.ckpt')
            initial_epoch = int(args.resume_ckpt)
            print(f'Resuming Training from {initial_epoch} epoch')
        gc.collect()
        print('Training Now')
        # Train the model
        model.fit(
            x=train_dataset,
            epochs=args.epochs,
            batch_size=args.bs,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            verbose=1,
            validation_data=val_dataset
        )
    elif args.test_vq:
        print(f'Testing Scaled VQVAE monai with ckpt - {args.suffix}-{args.test_epoch}')
        with strategy.scope():
            model = VQVAE(
                in_channels=1,
                out_channels=1,
                num_channels=(32, 64, 128),
                num_res_channels=(32, 64, 128),
                num_res_layers=3,
                downsample_parameters=((2, 4, 1, 'same'), (2, 4, 1, 'same'), (2, 4, 1, 'same')),
                upsample_parameters=((2, 4, 1, 'same', 0), (2, 4, 1, 'same', 0), (2, 4, 1, 'same')),
                num_embeddings=256,
                embedding_dim=64,
                num_gpus=args.num_gpus,
                kernel_resize=args.kernel_resize)

        model.load_weights(os.path.join(
            '/N/slate/aajais/checkpoints-vqvae-monai-scaled-128', args.suffix, str(args.test_epoch)+'.ckpt'))
        test_dataset = dataset.batch(args.bs).prefetch(
            tf.data.experimental.AUTOTUNE)

        directory = f'/N/slate/aajais/reconst-vqvae-monai-scaled-128/{args.suffix}/'
        if not os.path.exists(directory):
                os.makedirs(directory) 
        loss = []
        for i, (x, _) in tqdm(enumerate(test_dataset)):
            np.save(directory + f'{i}-original-{args.suffix}.npy', x.numpy())
            reconst = model(x)
            loss.append(tf.reduce_mean((reconst-x)**2))
            print(f'Test Loss is {sum(loss)/len(loss)}')
            np.save(directory + f'{i}-reconst3d-{args.suffix}-epoch{args.test_epoch}.npy', reconst.numpy())
    elif args.train_dm:
        dataset = dataset.batch(args.bs).prefetch(tf.data.experimental.AUTOTUNE)
    
        print(f'Training DM3D model with VQVAE ckpt - {args.vqvae_load_ckpt}')
        print('Training quantized latents')
        with strategy.scope():
            model = DiffusionModel(latent_size=int(
                64/4), num_embed=256, latent_channels=64, vqvae_load_ckpt=args.vqvae_load_ckpt, args=args)      
              
            model.compile(
                loss=keras.losses.MeanSquaredError(
                    reduction=tf.keras.losses.Reduction.SUM),
                optimizer=keras.optimizers.legacy.Adam(learning_rate=args.lr),
            )

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f'/N/slate/aajais/checkpoints-dm/{args.suffix}/'+'{epoch}.ckpt',
            save_weights_only=True,
            save_best_only=args.save_best_only)
        
        csv_logger = tf.keras.callbacks.CSVLogger(f'/N/slate/aajais/checkpoints-dm/{args.suffix}/training.log', append=True)

        if args.test_run:
                callbacks = []
        else:
            callbacks = [model_checkpoint_callback, csv_logger]

        initial_epoch = 0
        if args.resume_ckpt:
            model.load_weights(os.path.join(
            '/N/slate/aajais/checkpoints-dm', args.suffix, str(args.resume_ckpt)+'.ckpt'))
            initial_epoch = int(args.resume_ckpt)
            print(f'Resuming Training from {initial_epoch} epoch')

        print('Training Now')
        model.fit(
            dataset,
            epochs=args.epochs,
            batch_size=args.bs,
            callbacks=callbacks,
            initial_epoch=initial_epoch,
            verbose=1
        )
    elif args.test_dm:
        print(
            f'Testing Diffusion Model with ckpt - {args.suffix}-{args.test_epoch}')
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = DiffusionModel(latent_size=int(
                64/4), num_embed=256, latent_channels=64, vqvae_load_ckpt=args.vqvae_load_ckpt, args=args)

        model.load_weights(os.path.join(
            '/N/slate/aajais/checkpoints-dm', args.suffix, str(args.test_epoch)+'.ckpt'))
        args.suffix += 'epoch'+str(args.test_epoch)
        model.test(args.suffix)

if __name__=='__main__':
    print(tf.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--augment', default=False, action='store_true', help='Augment Data (FBC)')
    parser.add_argument('--train_vq', action='store_true', help='training flag - VQVAE')
    parser.add_argument('--train_dm', action='store_true', help='training flag - Diffsuion')
    parser.add_argument('--dataset', type=str, default='both', help='options for dataset -> HCP, NFBS, CC, BraTS, all-H, all-T')
    parser.add_argument('--test_vq', action='store_true', help='testing flag - VQVAE')
    parser.add_argument('--test_dm', action='store_true', help='testing flag - Diffsuion')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lbs', type=int, default=5, help='Batch size per gpu')
    parser.add_argument('--epochs', type=int, default=200, help='Epochs')
    parser.add_argument('--val_perc', type=float, default=0.1, help='Validation Percentage of Dataset')
    parser.add_argument('--suffix', default='basic', type=str, help='output or ckpts saved with this suffix')
    parser.add_argument('--num_gpus', default=2, type=int, help='Number of GPUs to be used')
    parser.add_argument('--kernel_resize', action='store_true', help='kernel resize flag')
    parser.add_argument('--test_epoch', type=int)
    parser.add_argument('--save_best_only', default=False, action='store_true')
    parser.add_argument('--vqvae_load_ckpt', type=str, default=None)
    parser.add_argument('--timesteps', type=int, default=300)
    parser.add_argument('--resume_ckpt', type=str)
    parser.add_argument('--test_run', default=False, action='store_true')
    args = parser.parse_args()

    run(args)
