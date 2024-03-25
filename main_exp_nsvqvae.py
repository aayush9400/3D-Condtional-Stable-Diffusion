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
from networks.nsvqvae import VQVAE, ReplaceCodebookCallback, WandbImageCallback

import wandb
from wandb.keras import WandbCallback

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
    wandb.init(project='nsvqvae')
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
    if args.test_vq:
        lis = dataset_list[-args.test_size :]
    print("Total images available for training: ", len(lis))
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )

    dataset_save_path = (
            f"/N/slate/aajais/skullstripping_datasets/training_data/B12-KR-AUG-all-T/"
        )
    dataset = load_dataset(dataset_save_path)
    dataset = dataset.take(3000)
    args.suffix += "-NSVQ-"
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

        # Use wandb.config to access the hyperparameters
        channel_list = eval(args.channel_list)  # Convert string to tuple
        num_embedding = args.num_embedding
        embedding_dim = args.embedding_dim

        # Dynamically set the number of residual layers or any other parameter based on channel_list
        num_res_layers = len(channel_list)  
        
        # downsample_parameters=(stride, kernel_size, dilation_rate, padding)
        downsample_parameters = [(2, 4, 1, "same") for _ in channel_list]  
        upsample_parameters = [(2, 4, 1, "same", 0) for _ in channel_list]  

        print(f"Training Scaled VQVAE monai")
        with strategy.scope():
            model = VQVAE(
                in_channels=1,
                out_channels=1,
                num_channels=channel_list,
                num_res_channels=channel_list,
                num_res_layers=num_res_layers,
                downsample_parameters=downsample_parameters,
                upsample_parameters=upsample_parameters,
                num_embeddings=num_embedding,
                embedding_dim=embedding_dim,
                num_gpus=args.num_gpus,
                kernel_resize=args.kernel_resize,
            )

            x = tf.keras.layers.Input(shape=(128, 128, 128, 1))
            m = tf.keras.Model(inputs=[x], outputs=model(x))
            print(m.summary())

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
                monitor="quantize_loss",
                factor=0.02,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            )

            wandbImage = WandbImageCallback(model, val_dataset, log_freq=10)
            replace_codebook_callback = ReplaceCodebookCallback(model.get_nsvq_model(), batch_size=args.bs, frequency=15)

            if args.test_run:
                callbacks = [reduce_lr, replace_codebook_callback]
            else:
                # callbacks = [model_checkpoint_callback, csv_logger, reduce_lr, replace_codebook_callback, WandbCallback(save_model=False)]
                callbacks = [reduce_lr, replace_codebook_callback, WandbCallback(save_model=False), wandbImage]
        initial_epoch = 0
        if args.resume_ckpt:
            model.load_weights(
                f"/N/slate/aajais/checkpoints-vqvae-monai-scaled-128/{args.suffix}/"
                + args.resume_ckpt
                + ".ckpt"
            )
            initial_epoch = int(args.resume_ckpt)
            print(f"Resuming Training from {initial_epoch} epoch")

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
        wandb.finish()
    elif args.test_vq:
        print(f"Testing Scaled VQVAE monai with ckpt - {args.suffix}-{args.test_epoch}")
        with strategy.scope():
            model = VQVAE(
                in_channels=1,
                out_channels=1,
                num_channels=(32, 64, 128, 256),
                num_res_channels=(32, 64, 128, 256),
                num_res_layers=4,
                # downsample_parameters=(stride, kernel_size, dilation_rate, padding)
                downsample_parameters=(
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
                ),
                num_embeddings=1024,
                embedding_dim=256,
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
            reconst, _ = model(x)
            loss.append(tf.reduce_mean((reconst - x) ** 2))
            print(f"Test Loss is {sum(loss)/len(loss)}")
            np.save(
                directory + f"{i}-reconst3d-{args.suffix}-epoch{args.test_epoch}.npy",
                reconst.numpy(),
            )

if __name__ == "__main__":
    print(tf.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--create_dataset", default=False, action="store_true", help="Create Dataset"
    )
    parser.add_argument(
        "--augment", default=False, action="store_true", help="Augment Data (F,B,C)"
    )
    parser.add_argument("--train_vq", default=True, action="store_true", help="training flag - VQVAE")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all-T",
        help="options for dataset -> HCP, NFBS, CC, BraTS, all, all-T",
    )
    parser.add_argument("--test_vq", default=False, action="store_true", help="testing flag - VQVAE")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--lbs", type=int, default=6, help="Batch size per gpu")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs")
    parser.add_argument(
        "--val_perc", type=float, default=0.1, help="Validation Percentage of Dataset"
    )
    parser.add_argument(
        "--suffix",
        default="wandb",
        type=str,
        help="output or ckpts saved with this suffix",
    )
    parser.add_argument(
        "--num_gpus", default=2, type=int, help="Number of GPUs to be used"
    )
    parser.add_argument(
        "--kernel_resize", default=True, action="store_true", help="kernel resize flag"
    )
    parser.add_argument("--test_epoch", type=int)
    parser.add_argument("--save_best_only", default=False, action="store_true")
    parser.add_argument("--vqvae_load_ckpt", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--resume_ckpt", type=str)
    parser.add_argument("--test_run", default=False, action="store_true")
    parser.add_argument("--channel_list", type=str, help="List of channels in the format '(num1,num2,...)'")
    parser.add_argument("--num_embedding", type=int, help="Number of embeddings")
    parser.add_argument("--embedding_dim", type=int, help="Dimension of embeddings")
    args = parser.parse_args()

    run(args)
