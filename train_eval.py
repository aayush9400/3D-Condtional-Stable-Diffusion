import os
import random
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from vqvae3d_monai import VQVAE
from dm3d import DiffusionModel

from data_preprocessing import datasetHelperFunc 
from dataset import get_dataset_list

def run(args):
    print(tf.__version__)

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
    if args.augment:
        args.suffix += '-AUG'
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
    print('Total Images in dataset: ', len(dataset_list))
    
    args.test_size = len(dataset_list) - (len(dataset_list)//args.bs)*args.bs
    print(args)

    lis = dataset_list[:-args.test_size] if args.test_size > 0 else dataset_list[:]
    if args.test:
        lis = dataset_list[-args.test_size:]

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    dataset = tf.data.Dataset.from_tensor_slices(lis)
    if args.train:
        dataset = dataset.shuffle(len(lis), reshuffle_each_iteration=True)
    dataset = dataset.map(lambda x: tf.numpy_function(func=datasetHelperFunc, inp=[x], Tout=[tf.float32, tf.float32]),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    if args.augment and args.train:
        augmented_dataset = tf.data.Dataset.from_tensor_slices(random.sample(lis, int(len(lis)*0.03)*args.bs))
        augmented_dataset = augmented_dataset.map(
            lambda x: tf.numpy_function(func=datasetHelperFunc, inp=[x], Tout=[tf.float32, tf.float32]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        print("Number of augmented images: ", augmented_dataset.cardinality().numpy())
        dataset = dataset.concatenate(augmented_dataset)
        print("Number of total images: ", dataset.cardinality().numpy())

    if args.train:
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
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            )

            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=f'./checkpoints-vqvae-monai-scaled-128/{args.suffix}/'+'{epoch}.ckpt',
                save_weights_only=True,
                save_best_only=args.save_best_only)
            
            csv_logger = tf.keras.callbacks.CSVLogger(f'./checkpoints-vqvae-monai-scaled-128/{args.suffix}/training.log', append=True)

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
                f'./checkpoints-vqvae-monai-scaled-128/{args.suffix}/'+args.resume_ckpt+'.ckpt')
            initial_epoch = int(args.resume_ckpt)
            print(f'Resuming Training from {initial_epoch} epoch')

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

    elif args.test:
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

    elif args.train_dm:
        dataset = dataset.batch(args.bs).prefetch(tf.data.experimental.AUTOTUNE)
    
        print(f'Training DM3D model with VQVAE ckpt - {args.vqvae_load_ckpt}')
        print('Training quantized latents')
        with strategy.scope():
            model = DiffusionModel(latent_size=int(
                128/4), num_embed=256, latent_channels=64, vqvae_load_ckpt=args.vqvae_load_ckpt, args=args)      
              
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.SUM),
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        )

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f'./checkpoints-dm/{args.suffix}/'+'{epoch}.ckpt',
            save_weights_only=True)
        
        csv_logger = tf.keras.callbacks.CSVLogger(f'./checkpoints-dm/{args.suffix}/training.log', append=True)

        model.fit(
            dataset,
            epochs=args.epochs,
            batch_size=args.bs,
            callbacks=[csv_logger, model_checkpoint_callback],
            verbose=1
        )

    elif args.test_dm:
        print(
            f'Testing Diffusion Model with ckpt - {args.suffix}-{args.test_epoch}')
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = DiffusionModel(latent_size=int(
                128/4), num_embed=256, latent_channels=64, vqvae_load_ckpt=args.vqvae_load_ckpt, args=args)

        model.load_weights(os.path.join(
            './checkpoints-dm', args.suffix, str(args.test_epoch)+'.ckpt'))
        args.suffix += 'epoch'+str(args.test_epoch)
        model.test(args.suffix)    