import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
# import tensorflow_probability as tfp
import tensorflow as tf

from dipy.io.image import load_nifti
from dipy.align.reslice import reslice
from scipy.ndimage import affine_transform

from vqvae3d_monai import VQVAE
from dm3d import DiffusionModel

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true')
parser.add_argument('--train_dm', action='store_true')
parser.add_argument('--cc359', action='store_true')
parser.add_argument('--both_datasets', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--test_dm', action='store_true')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--lbs', type=int, default=5, help='Batch size per gpu')
parser.add_argument('--epochs', type=int, default=200, help='Epochs')
parser.add_argument('--suffix', default='basic', type=str,
                    help='output or ckpts saved with this suffix')
parser.add_argument('--num_gpus', default=2, type=int)
parser.add_argument('--kernel_resize', action='store_true')
parser.add_argument('--test_epoch', type=int)
parser.add_argument('--save_best_only', action='store_true')
parser.add_argument('--vqvae_load_ckpt', type=str, default=None)
parser.add_argument('--timesteps', type=int, default=300)
parser.add_argument('--resume_ckpt', type=str)
args = parser.parse_args()


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


def transform_img_cc359(image, affine, voxsize, output_shape=(128, 128, 128)):
    s = image.shape
    a, b, c = s[0]/output_shape[0], s[1]/output_shape[1], s[2]/output_shape[2]
    image, affine = reslice(image, affine, voxsize, (a, b, c))
    # print(image)
    return image, affine


dataset_list = []

files = ['./dataset.txt']
if args.cc359:
    files = ['./dataset_cc359.txt']
elif args.both_datasets:
    files.append('./dataset_cc359.txt')

for file in files:
    with open(file, 'r') as f:
        lin = f.readline()
        while(lin):
            dataset_list.append(lin[:-1])
            lin = f.readline()
print(len(dataset_list))


def datasetHelperFunc(path):
    transform_vol, mask = None, None
    if isinstance(path, bytes):
        path = str(path.decode('utf-8'))

    if 'CC' in path:
        vol, affine, voxsize = load_nifti(
            '/N/slate/varbayan/CC359/Original/'+path, return_voxsize=True)
        mask, _ = load_nifti(
            '/N/slate/varbayan/CC359/STAPLE/'+path[:-7]+'_staple.nii.gz')
        mask[mask < 1] = 0  # Values <1 in the mask is background
        vol = vol*mask
    else:
        #vol, affine, voxsize = load_nifti(str(path.decode('utf-8')), return_voxsize=True)
        vol, affine, voxsize = load_nifti(path, return_voxsize=True)
        mask, _ = load_nifti(path[:-7]+'mask.nii.gz')
        mask[mask < 1] = 0  # Values <1 in the mask is background
        vol = vol*mask

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
if args.cc359:
    args.suffix = 'CC-'+args.suffix
elif args.both_datasets:
    args.suffix += '-both'
if args.train_dm or args.test_dm:
    args.suffix += '-DM'+str(args.timesteps)
print('Global Batch Size: ', args.bs)
args.test_size = len(dataset_list) - (len(dataset_list)//args.bs)*args.bs
print(args)

lis = dataset_list[:-args.test_size] if args.test_size > 0 else dataset_list[:]
if args.test or args.test_dm:
    lis = dataset_list[-args.test_size:]


dataset = tf.data.Dataset.from_tensor_slices(lis)
if args.train or args.train_dm:
    dataset = dataset.shuffle(len(lis), reshuffle_each_iteration=True)
dataset = dataset.map(lambda x: tf.numpy_function(func=datasetHelperFunc, inp=[x], Tout=[tf.float32, tf.float32]),
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

# def data_generator():
#     for file_path in lis:
#         # Load and preprocess a single sample
#         data = datasetHelperFunc(file_path)
#         # Yield the processed data as a batch
#         if data[0] is None:
#             continue
#         yield (data[0], data[1])

# dataset = tf.data.Dataset.from_generator(data_generator, output_signature=(
#          tf.TensorSpec(shape=(128,128,128,1), dtype=tf.float32),
#          tf.TensorSpec(shape=(128,128,128,1), dtype=tf.float32))).shuffle(125, reshuffle_each_iteration=True)

# test_dataset = tf.data.Dataset.from_generator(data_generator, output_types=tf.float32)


if args.train:
    print(f'Training VQVAE monai')
    dataset = dataset.batch(args.bs).prefetch(tf.data.experimental.AUTOTUNE)
    with strategy.scope():
        model = VQVAE(
            in_channels=1,
            out_channels=1,
            num_channels=(32, 64),
            num_res_channels=(32, 64),
            num_res_layers=2,
            downsample_parameters=((2, 4, 1, 'same'), (2, 4, 1, 'same')),
            upsample_parameters=((2, 4, 1, 'same', 0), (2, 4, 1, 'same', 0)),
            num_embeddings=256,
            embedding_dim=32,
            num_gpus=float(args.num_gpus),
            kernel_resize=args.kernel_resize)

        x = tf.keras.layers.Input(shape=(128, 128, 128, 1))
        m = tf.keras.Model(inputs=[x], outputs=model(x))
        print(m.summary())

        # Compile the model
        model.compile(
            # loss=keras.losses.MeanSquaredError(tf.keras.losses.Reduction.SUM),
            optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        )

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f'./checkpoints-vqvae-monai/{args.suffix}/' +
            '{epoch}.ckpt',
            save_weights_only=True,
            save_best_only=args.save_best_only)

    initial_epoch = 0
    if args.resume_ckpt:
        model.load_weights(
            f'./checkpoints-vqvae-monai/{args.suffix}/'+args.resume_ckpt+'.ckpt')
        initial_epoch = int(args.resume_ckpt)
        print(f'Resuming Trainig from {initial_epoch} epoch')

    print('Training Now')
    # Train the model
    model.fit(
        x=dataset,
        epochs=args.epochs,
        batch_size=args.bs,
        initial_epoch=initial_epoch,
        callbacks=[model_checkpoint_callback]
    )

elif args.test:
    print(f'Testing VQVAE monai with ckpt - {args.suffix}-{args.test_epoch}')
    with strategy.scope():
        model = VQVAE(
            in_channels=1,
            out_channels=1,
            num_channels=(32, 64),
            num_res_channels=(32, 64),
            num_res_layers=2,
            downsample_parameters=((2, 4, 1, 'same'), (2, 4, 1, 'same')),
            upsample_parameters=((2, 4, 1, 'same', 0), (2, 4, 1, 'same', 0)),
            num_embeddings=256,
            embedding_dim=32,
            num_gpus=float(args.num_gpus),
            kernel_resize=args.kernel_resize)

    model.load_weights(os.path.join(
        '/N/slate/varbayan/checkpoints-vqvae-monai', args.suffix, str(args.test_epoch)+'.ckpt'))
    test_dataset = dataset.batch(args.bs).prefetch(
        tf.data.experimental.AUTOTUNE).take(1)

    for x, _ in test_dataset:
        np.save(
            f'./reconst_vqvae3d_monai/original-{args.suffix}.npy', x.numpy())
        reconst = model(x)
        loss = tf.reduce_mean((reconst-x)**2)
        print(f'Test Loss is {loss}')
        np.save(
            f'./reconst_vqvae3d_monai/reconst3d-{args.suffix}-epoch{args.test_epoch}.npy', reconst.numpy())


elif args.train_dm:
    print(f'Training DM3D model with VQVAE ckpt - {args.vqvae_load_ckpt}')
    print('Training quantized latents')
    with strategy.scope():
        model = DiffusionModel(latent_size=int(
            128/4), num_embed=256, latent_channels=32, vqvae_load_ckpt=args.vqvae_load_ckpt, args=args)

    dataset = dataset.batch(args.bs).prefetch(tf.data.experimental.AUTOTUNE)
    model.compile(
        loss=keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.SUM),
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'./checkpoints-dm/{args.suffix}/'+'{epoch}.ckpt',
        save_weights_only=True)

    model.fit(
        dataset,
        epochs=args.epochs,
        batch_size=args.bs,
        callbacks=[model_checkpoint_callback]
        # callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images)],
    )

elif args.test_dm:
    print(
        f'Testing Diffusion Model with ckpt - {args.suffix}-{args.test_epoch}')
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = DiffusionModel(latent_size=int(
            128/4), num_embed=256, latent_channels=32, vqvae_load_ckpt=args.vqvae_load_ckpt, args=args)

    model.load_weights(os.path.join(
        '/N/slate/varbayan/checkpoints-dm', args.suffix, str(args.test_epoch)+'.ckpt'))
    args.suffix += 'epoch'+str(args.test_epoch)
    model.test(args.suffix)
