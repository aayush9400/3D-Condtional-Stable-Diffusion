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
from networks.vqvae3d_monai import VQVAE
from networks.conditional_dm3d import DiffusionModel, WandbImageCallback

import wandb
from wandb.keras import WandbCallback
# from dm3d import DiffusionModel

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
    wandb.init(project="CLDM", config=args.__dict__, log_graph=False)

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
    if args.test_dm:
        lis = dataset_list[-args.test_size :]
    print("Total images available for training: ", len(lis))

    dataset_save_path = (
            f"/N/slate/aajais/skullstripping_datasets/training_data/with_mask_context_{args.suffix}/"
        )
    dataset = load_dataset(dataset_save_path)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )
    args.suffix += "_cldm"
    if args.train_dm:
        dataset = dataset.batch(args.bs).prefetch(tf.data.experimental.AUTOTUNE)
        dataset = dataset.__iter__()
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

        wandb_callback = WandbCallback(save_model=False, log_weights=False, log_model=False, log_graph=False)

        if args.test_run:
            callbacks = []
        else:
            callbacks = [wandb_callback, WandbImageCallback(model)]

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
        wandb.finish()
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
        "--context_flag", default=False, action="store_true", help="Context Flag for Diffusion Training"
    )
    parser.add_argument(
        "--augment", default=False, action="store_true", help="Augment Data (F,B,C)"
    )
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
