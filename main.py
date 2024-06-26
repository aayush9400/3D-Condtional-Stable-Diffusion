import time
import argparse
import os
import gc
import glob
import numpy as np
import random
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

from dataset_utils import create_dataset, load_dataset
from networks.vqvae3d_monai import VQVAE, ReplaceCodebookCallback
from networks.dm3d import DiffusionModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1 = INFO, 2 = WARNING, 3 = ERROR)
tf.get_logger().setLevel('ERROR')


def get_dataset_list(dataset_dir="/N/slate/aajais/skullstripping_datasets/"):
    dataset_list = []

    if args.dataset == "CC":
        dataset_list = glob.glob(
            os.path.join(dataset_dir, "CC359", "Original", "*.nii.gz")
        )
    elif args.dataset == "NFBS":
        dataset_list = glob.glob(
            os.path.join(
                dataset_dir, "NFBS_Dataset", "*", "sub-*_ses-NFB3_T1w_brain.nii.gz"
            )
        )
    elif args.dataset == "HCP":
        dataset_list = glob.glob(os.path.join(dataset_dir, "HCP_T1", "T1", "*.nii.gz"))
    elif args.dataset == "BraTS":
        dataset_list = glob.glob(
            os.path.join(dataset_dir, "BraTS2021", "*", "*_t1.nii.gz")
        )
    elif args.dataset == "all":
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
    elif args.dataset == "all-T":
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

    if args.test_run:
        print(len(dataset_list))
        # dataset_list = dataset_list[:240]

    return dataset_list


def run(args):
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
    args.num_gpus = strategy.num_replicas_in_sync
    print(f"Number of devices: {args.num_gpus}")
    gpus = tf.config.experimental.list_physical_devices("GPU")
    print(gpus)
    print(tf.config.list_logical_devices())

    args.bs = args.lbs * args.num_gpus
    args.suffix = "B" + str(args.bs)

    if args.kernel_resize:
        args.suffix += "-KR"
    if args.augment:
        args.suffix += "-AUG"
    if args.dataset == "CC":
        args.suffix = "CC-" + args.suffix
    elif args.dataset == "NFBS":
        args.suffix = "NFBS-" + args.suffix
    elif args.dataset == "HCP":
        args.suffix = "HCP-" + args.suffix
    elif args.dataset == "BraTS":
        args.suffix = "BraTS-" + args.suffix
    elif args.dataset == "all":
        args.suffix += "-all"
    elif args.dataset == "all-T":
        args.suffix += "-all-T"
    print("Global Batch Size: ", args.bs)

    dataset_list = get_dataset_list()
    print("Total Images in dataset: ", len(dataset_list))
    args.test_size = len(dataset_list) - (len(dataset_list) // args.bs) * args.bs
    print(args)

    lis = dataset_list[: -args.test_size] if args.test_size > 0 else dataset_list[:]
    if args.test_vq or args.test_dm:
        lis = dataset_list[-args.test_size :]
    print("Total images available for training: ", len(lis))

    dataset_save_path = (
        f"/N/slate/aajais/skullstripping_datasets/training_data/with_mask_context_{args.suffix}/"
    )
    if args.create_dataset:
        start = time.time()
        # print(start)
        create_flag = create_dataset(
            dataset_list=lis,
            batch_size=args.bs,
            dataset_save_path=dataset_save_path,
            augment_flag=args.augment,
            save_flag=args.create_dataset,
        )
        end = time.time()
        print(create_flag, "time taken:", (end - start) / 60)
        dataset = load_dataset(dataset_save_path)
    else:
        if args.train_vq or args.train_dm:
            if os.path.exists(dataset_save_path):
                dataset = load_dataset(dataset_save_path)
            else:
                dataset = create_dataset(dataset_list=lis, 
                                        batch_size=args.bs, 
                                        dataset_save_path=dataset_save_path, 
                                        augment_flag=args.augment,
                                        save_flag=args.create_dataset)
        else:
            dataset = create_dataset(
                dataset_list=lis,
                batch_size=args.bs,
                dataset_save_path=dataset_save_path,
                augment_flag=args.augment,
                save_flag=args.create_dataset,
            )
    # dataset = dataset.take(3000)
    dataset = dataset.shuffle(buffer_size = 2 * args.lbs) 
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )
    args.suffix += f"-{args.exp_name}-"
    if args.train_vq:
        train_size = int((1 - args.val_perc) * dataset.cardinality().numpy())
        train_size = train_size - (train_size % args.bs)

        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size)

        train_dataset = train_dataset.batch(args.bs).prefetch(
            tf.data.experimental.AUTOTUNE
        )
        val_dataset = val_dataset.batch(args.bs).prefetch(tf.data.experimental.AUTOTUNE)

        train_dataset = train_dataset.with_options(options)
        val_dataset = val_dataset.with_options(options)

        train_dataset_cardinality = train_dataset.cardinality().numpy()
        val_dataset_cardinality = val_dataset.cardinality().numpy()

        print(
            f"Number of images in the training dataset: {train_dataset_cardinality * args.bs}"
        )
        print(
            f"Number of images in the validation dataset: {val_dataset_cardinality * args.bs}"
        )

        print(f"Training Scaled VQVAE monai")
        with strategy.scope():
            model = VQVAE(
                in_channels=1,
                out_channels=1,
                num_channels=(32,64,128),
                num_res_channels=(32,64,128),
                num_res_layers=3,
                # downsample_parameters=(stride, kernel_size, dilation_rate, padding)
                downsample_parameters=(
                    (2, 4, 1, "same"),
                    (2, 4, 1, "same"),
                    (2, 4, 1, "same"),
					# (2, 4, 1, "same"),
                    # (2, 4, 1, "same"),
                ),
                upsample_parameters=(
                    (2, 4, 1, "same", 0),
                    (2, 4, 1, "same", 0),
                    (2, 4, 1, "same", 0),
					# (2, 4, 1, "same", 0),
                    # (2, 4, 1, "same", 0),
                ),
                num_embeddings=512,
                embedding_dim=256,
                num_gpus=args.num_gpus,
                kernel_resize=args.kernel_resize,
            )

            x = tf.keras.layers.Input(shape=(128, 128, 128, 1))
            m = tf.keras.Model(inputs=[x], outputs=model(x))
            print(m.summary())

            # Compile the model
            # Loss is implemented in model file
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=args.lr),
            )

            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=f"/N/slate/aajais/checkpoints-vqvae-monai-scaled-128/{args.suffix}/"
                + "{epoch}.ckpt",
                save_weights_only=True,
                save_best_only=args.save_best_only,
            )

            csv_logger = tf.keras.callbacks.CSVLogger(
                f"/N/slate/aajais/checkpoints-vqvae-monai-scaled-128/{args.suffix}/training.log",
                append=True,
            )

            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="loss",
                factor=0.02,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            )

            replace_codebook_callback = ReplaceCodebookCallback(model.get_vq_model(), batch_size=args.bs, frequency=10)

            if args.test_run:
                callbacks = [reduce_lr]
            else:
                callbacks = [model_checkpoint_callback, csv_logger, reduce_lr, replace_codebook_callback]

        initial_epoch = 0
        if args.resume_ckpt:
            model.load_weights(
                f"/N/slate/aajais/checkpoints-vqvae-monai-scaled-128/{args.suffix}/"
                + args.resume_ckpt
                + ".ckpt"
            )
            initial_epoch = int(args.resume_ckpt)
            print(f"Resuming Training from {initial_epoch} epoch")
        gc.collect()
        print("Training Now")
        # Train the model
        model.fit(
            x=train_dataset,
            epochs=args.epochs,
            batch_size=args.bs,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            verbose=1,
            validation_data=val_dataset,
        )
    elif args.test_vq:
        print(f"Testing Scaled VQVAE monai with ckpt - {args.suffix}-{args.test_epoch}")
        with strategy.scope():
			#best
            # model = VQVAE(
            #     in_channels=1,
            #     out_channels=1,
            #     num_channels=(32, 64, 128),
            #     num_res_channels=(32, 64, 128),
            #     num_res_layers=3,
            #     downsample_parameters=(
            #         (2, 4, 1, "same"),
            #         (2, 4, 1, "same"),
            #         (2, 4, 1, "same"),
            #     ),
            #     upsample_parameters=(
            #         (2, 4, 1, "same", 0),
            #         (2, 4, 1, "same", 0),
            #         (2, 4, 1, "same",0),
            #     ),
            #     num_embeddings=256,
            #     embedding_dim=64,
            #     num_gpus=args.num_gpus,
            #     kernel_resize=args.kernel_resize,
            # )
			#dm
#             model = VQVAE(
#                 in_channels=1,
#                 out_channels=1,
#                 num_channels=(32, 64, 128,256),
#                 num_res_channels=(32, 64, 128,256),
#                 num_res_layers=5,
#                 # downsample_parameters=(stride, kernel_size, dilation_rate, padding)
#                 downsample_parameters=(
#                     (2, 4, 1, "same"),
#                     (2, 4, 1, "same"),
#                     (2, 4, 1, "same"),
# 					(2, 4, 1, "same"),
#                 ),
#                 upsample_parameters=(
#                     (2, 4, 1, "same", 0),
#                     (2, 4, 1, "same", 0),
#                     (2, 4, 1, "same", 0),
# 					(2, 4, 1, "same", 0),
#                 ),
#                 num_embeddings=1024,
#                 embedding_dim=256,
#                 num_gpus=args.num_gpus,
#                 kernel_resize=args.kernel_resize,
#             )
			#new
            model = VQVAE(
                in_channels=1,
                out_channels=1,
                num_channels=(32, 64, 128,256,512),
                num_res_channels=(32, 64, 128,256,512),
                num_res_layers=5,
                # downsample_parameters=(stride, kernel_size, dilation_rate, padding)
                downsample_parameters=(
                    (2, 4, 1, "same"),
                    (2, 4, 1, "same"),
                    (2, 4, 1, "same"),
                    (2, 4, 1, "same"),
                    (2, 4, 1, "same"),
                ),
                upsample_parameters=(
                    (2, 4, 1, "same", 0),
                    (2, 4, 1, "same", 0),
                    (2, 4, 1, "same", 0),
                    (2, 4, 1, "same", 0),
                    (2, 4, 1, "same", 0),
                ),
                num_embeddings=1024,
                embedding_dim=512,
                num_gpus=args.num_gpus,
                kernel_resize=args.kernel_resize,
            )
        model.load_weights(
            os.path.join(
                "/N/slate/aajais/checkpoints-vqvae-monai-scaled-128",
                args.suffix,
                str(args.test_epoch) + ".ckpt",
            )
        )
        test_dataset = dataset.batch(args.bs).prefetch(tf.data.experimental.AUTOTUNE)

        directory = f"/N/slate/aajais/reconst-vqvae-monai-scaled-128/{args.suffix}/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        loss = []
        for i, (x, _, _) in tqdm(enumerate(test_dataset)):
            np.save(directory + f"{i}-original-{args.suffix}.npy", x.numpy())
            reconst = model(x)
            loss.append(tf.reduce_mean((reconst - x) ** 2))
            print(f"Test Loss is {sum(loss)/len(loss)}")
            np.save(
                directory + f"{i}-reconst3d-{args.suffix}-epoch{args.test_epoch}.npy",
                reconst.numpy(),
            )
    elif args.train_dm:
        dataset = dataset.batch(args.bs).prefetch(tf.data.experimental.AUTOTUNE)

        print(f"Training DM3D model with VQVAE ckpt - {args.vqvae_load_ckpt}")
        print("Training quantized latents")
        with strategy.scope():
            model = DiffusionModel(
                latent_size=int(64 / 8),
                num_embed=1024,
                latent_channels=256,
                vqvae_load_ckpt=args.vqvae_load_ckpt,
                args=args,
            )

            model.compile(
                loss=keras.losses.MeanSquaredError(
                    reduction=tf.keras.losses.Reduction.SUM
                ),
                optimizer=keras.optimizers.Adam(learning_rate=args.lr),
            )

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"/N/slate/aajais/checkpoints-dm/{args.suffix}/" + "{epoch}.ckpt",
            save_weights_only=True,
            save_best_only=args.save_best_only,
        )

        csv_logger = tf.keras.callbacks.CSVLogger(
            f"/N/slate/aajais/checkpoints-dm/{args.suffix}/training.log", append=True
        )

        if args.test_run:
            callbacks = []
        else:
            callbacks = [model_checkpoint_callback, csv_logger]

        initial_epoch = 0
        if args.resume_ckpt:
            model.load_weights(
                os.path.join(
                    "/N/slate/aajais/checkpoints-dm",
                    args.suffix,
                    str(args.resume_ckpt) + ".ckpt",
                )
            )
            initial_epoch = int(args.resume_ckpt)
            print(f"Resuming Training from {initial_epoch} epoch")

        print("Training Now")
        model.fit(
            dataset,
            epochs=args.epochs,
            batch_size=args.bs,
            callbacks=callbacks,
            initial_epoch=initial_epoch,
            verbose=1,
        )
    elif args.test_dm:
        print(f"Testing Diffusion Model with ckpt - {args.suffix}-{args.test_epoch}")
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = DiffusionModel(
                latent_size=int(64 / 4),
                num_embed=256,
                latent_channels=64,
                vqvae_load_ckpt=args.vqvae_load_ckpt,
                args=args,
            )

        model.load_weights(
            os.path.join(
                "/N/slate/aajais/checkpoints-dm",
                args.suffix,
                str(args.test_epoch) + ".ckpt",
            )
        )
        args.suffix += "epoch" + str(args.test_epoch)
        model.test(args.suffix)


if __name__ == "__main__":
    print(tf.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--create_dataset", default=False, action="store_true", help="Create Dataset"
    )
    parser.add_argument(
        "--augment", default=False, action="store_true", help="Augment Data (F,B,C)"
    )
    parser.add_argument("--train_vq", action="store_true", help="training flag - VQVAE")
    parser.add_argument(
        "--train_dm", action="store_true", help="training flag - Diffusion"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="both",
        help="options for dataset -> HCP, NFBS, CC, BraTS, all, all-T",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="mask",
    )
    parser.add_argument("--test_vq", action="store_true", help="testing flag - VQVAE")
    parser.add_argument(
        "--test_dm", action="store_true", help="testing flag - Diffusion"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--lbs", type=int, default=5, help="Batch size per gpu")
    parser.add_argument("--epochs", type=int, default=200, help="Epochs")
    parser.add_argument(
        "--val_perc", type=float, default=0.1, help="Validation Percentage of Dataset"
    )
    parser.add_argument(
        "--suffix",
        default="basic",
        type=str,
        help="output or ckpts saved with this suffix",
    )
    parser.add_argument(
        "--num_gpus", default=2, type=int, help="Number of GPUs to be used"
    )
    parser.add_argument(
        "--kernel_resize", action="store_true", help="kernel resize flag"
    )
    parser.add_argument("--test_epoch", type=int)
    parser.add_argument("--save_best_only", default=False, action="store_true")
    parser.add_argument("--vqvae_load_ckpt", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--resume_ckpt", type=str)
    parser.add_argument("--test_run", default=False, action="store_true")
    args = parser.parse_args()

    run(args)
