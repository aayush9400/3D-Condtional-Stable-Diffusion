import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from tensorflow import keras
from tensorflow.keras import layers
# import tensorflow_probability as tfp
import tensorflow as tf

from dipy.io.image import load_nifti
from dipy.align.reslice import reslice
from scipy.ndimage import affine_transform

from vqvae3d_monai import VQVAE
from dm3d import DiffusionModel


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


def datasetHelperFunc(path):
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
    return tf.convert_to_tensor(transform_vol, tf.float32), tf.convert_to_tensor(mask, tf.float32)


def get_dataset_list(dataset_dir='/N/slate/aajais/skullstripping_datasets/'):
    dataset_list = []

    if args.dataset == 'CC':
        dataset_list = glob.glob(os.path.join(dataset_dir, 'CC359', 'Original', '*.nii.gz'))
    elif args.dataset == 'NFBS':
        with open('./dataset_NFBS.txt', 'r') as fd:
            for row in fd:
                dataset_list.append(dataset_dir + row.split('\n')[0])
    elif args.dataset == 'HCP':
        dataset_list = glob.glob(os.path.join(dataset_dir, 'HCP_T1', 'T1', '*.nii.gz'))
    elif args.dataset == 'both':
        dataset_list = glob.glob(os.path.join(dataset_dir, 'CC359', 'Original', '*.nii.gz'))
        with open('./dataset_NFBS.txt', 'r') as fd:
            for row in fd:
                dataset_list.append(dataset_dir + row.split('\n')[0])
    elif args.dataset == 'all':
        dataset_list = glob.glob(os.path.join(dataset_dir, 'CC359', 'Original', '*.nii.gz'))
        with open('./dataset_NFBS.txt', 'r') as fd:
            for row in fd:
                dataset_list.append(dataset_dir + row.split('\n')[0])
        dataset_list.extend(glob.glob(os.path.join(dataset_dir, 'HCP_T1', 'T1', '*.nii.gz')))

    print('Total Images in dataset: ', len(dataset_list))
    return dataset_list


def run(args):
    strategy = tf.distribute.MirroredStrategy()
    args.num_gpus = strategy.num_replicas_in_sync
    print(f'Number of devices: {args.num_gpus}')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    print(tf.config.list_logical_devices())
        
    args.bs = args.lbs*args.num_gpus
    args.suffix = 'B'+str(args.bs)

    if args.kernel_resize:
        args.suffix += '-KR'
    if args.dataset=='CC':
        args.suffix = 'CC-'+args.suffix
    elif args.dataset=='NFBS':
        args.suffix = 'NFBS-'+args.suffix
    elif args.dataset=='HCP':
        args.suffix = 'HCP-'+args.suffix
    elif args.dataset == 'both':
        args.suffix += '-both'
    elif args.dataset == 'all':
        args.suffix += '-all'
    print('Global Batch Size: ', args.bs)

    dataset_list = get_dataset_list()
    
    args.test_size = len(dataset_list) - (len(dataset_list)//args.bs)*args.bs
    print(args)

    lis = dataset_list[:-args.test_size] if args.test_size > 0 else dataset_list[:]

    if args.train:
        dataset = tf.data.Dataset.from_tensor_slices(lis)
        dataset = dataset.shuffle(len(lis), reshuffle_each_iteration=True)
        dataset = dataset.map(lambda x: tf.numpy_function(func=datasetHelperFunc, inp=[x], Tout=[tf.float32, tf.float32]),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

        train_size = int((1 - args.val_perc) * len(lis))

        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size)

        print(f'Training Scaled VQVAE monai')
        train_dataset = train_dataset.batch(args.bs).prefetch(tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.batch(args.bs).prefetch(tf.data.experimental.AUTOTUNE)

        with strategy.scope():
            model = VQVAE(
                in_channels=1,
                out_channels=1,
                num_channels=(32, 64, 128),
                num_res_channels=(32, 64, 128),
                num_res_layers=3,
                downsample_parameters=((2, 4, 1, 'same'), (2, 4, 1, 'same'), (2, 4, 1, 'same')),
                upsample_parameters=((2, 4, 1, 'same', 0), (2, 4, 1, 'same', 0), (2, 4, 1, 'same', 0)),
                num_embeddings=256,
                embedding_dim=64,
                num_gpus=float(args.num_gpus),
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
                filepath=f'./checkpoints-vqvae-monai-scaled-128/{args.suffix}/'+'{epoch}.ckpt',
                save_weights_only=True,
                save_best_only=args.save_best_only)
            
            csv_logger = tf.keras.callbacks.CSVLogger(f'./checkpoints-vqvae-monai-scaled-128/{args.suffix}/training.log', append=True)

            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.1, 
                              patience=5, 
                              min_delta=1e-4, 
                              cooldown=2, 
                              min_lr=1e-6, 
                              verbose=1)
            
            callbacks = [model_checkpoint_callback, csv_logger, reduce_lr]

        initial_epoch = 0
        if args.resume_ckpt:
            model.load_weights(
                f'./checkpoints-vqvae-monai-scaled-128/{args.suffix}/'+args.resume_ckpt+'.ckpt')
            initial_epoch = int(args.resume_ckpt)
            print(f'Resuming Training from {initial_epoch} epoch')

        print('Training Now')
        # Train the model
        history = model.fit(
            x=train_dataset,
            epochs=args.epochs,
            batch_size=args.bs,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            verbose=1,
            validation_data=val_dataset
        )

    elif args.test:
        lis = dataset_list[-args.test_size:]
        dataset = tf.data.Dataset.from_tensor_slices(lis)
        dataset = dataset.map(lambda x: tf.numpy_function(func=datasetHelperFunc, inp=[x], Tout=[tf.float32, tf.float32]),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

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
                num_gpus=float(args.num_gpus),
                kernel_resize=args.kernel_resize)

        model.load_weights(os.path.join(
            './checkpoints-vqvae-monai-scaled-128', args.suffix, str(args.test_epoch)+'.ckpt'))
        test_dataset = dataset.batch(args.bs).prefetch(
            tf.data.experimental.AUTOTUNE)

        directory = f'./reconst-vqvae-monai-scaled-128/{args.suffix}/'
        if not os.path.exists(directory):
                os.makedirs(directory) 
        loss = []
        for i, (x, _) in tqdm(enumerate(test_dataset)):
            np.save(directory + f'{i}-original-{args.suffix}.npy', x.numpy())
            reconst = model(x)
            loss.append(tf.reduce_mean((reconst-x)**2))
            print(f'Test Loss is {sum(loss)/len(loss)}')
            np.save(directory + f'{i}-reconst3d-{args.suffix}-epoch{args.test_epoch}.npy', reconst.numpy())

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='training flag - VQVAE')
    parser.add_argument('--train_dm', action='store_true', help='training flag - Diffsuion')
    parser.add_argument('--dataset', type=str, default='both', help='options for dataset -> HCP, NFBS, CC, both, all')
    parser.add_argument('--test', action='store_true', help='testing flag - VQVAE')
    parser.add_argument('--test_dm', action='store_true', help='testing flag - Diffsuion')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
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
    args = parser.parse_args()

    run(args)