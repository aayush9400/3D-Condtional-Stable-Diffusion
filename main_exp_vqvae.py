import time
import argparse
import os
import gc
import glob
import numpy as np
import random
from tqdm import tqdm
import pandas as pd

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

from dataset_utils import create_dataset, load_dataset
from networks.vqvae3d_monai import VQVAE, ReplaceCodebookCallback
from training_utils import WandbImageCallback
from networks.dm3d import DiffusionModel

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


def build_and_compile_model(channel_list, num_embedding, embedding_dim, args, strategy):
    with strategy.scope():
        # Dynamically set the number of residual layers or any other parameter based on channel_list
        num_res_layers = len(channel_list)  
        
        # Dynamically construct downsample and upsample parameters
        downsample_parameters = [(2, 4, 1, "same") for _ in channel_list]  
        upsample_parameters = [(2, 4, 1, "same", 0) for _ in channel_list]  

        model = VQVAE(
            in_channels=2,
            out_channels=2,
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

        x = tf.keras.layers.Input(shape=(128, 128, 128, 2))
        m = tf.keras.Model(inputs=[x], outputs=model(x))
        print(m.summary())
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr))

    return model


def run_experiment(args, strategy, train_dataset, val_dataset):
    run_name = f"{args.channel_list}|{args.num_embedding}x{args.embedding_dim}|{args.bs}" 
    wandb.init(project='vqvae', entity='dipy_genai', name=run_name)
    # Use wandb.config to access the hyperparameters
    channels = eval(args.channel_list)  # Convert string to tuple
    num_embedding = args.num_embedding
    embedding_dim = args.embedding_dim

    model = build_and_compile_model(channels, num_embedding, embedding_dim, args, strategy)
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"/N/slate/aajais/checkpoints-vqvae-monai-scaled-128/{args.suffix}/"
        + "{epoch}.ckpt",
        save_weights_only=True,
        save_best_only=args.save_best_only,
        period=10
    )

    csv_logger = tf.keras.callbacks.CSVLogger(
        f"/N/slate/aajais/checkpoints-vqvae-monai-scaled-128/{args.suffix}/training.log",
        append=True,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss",
        factor=0.02,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )
    
    replace_codebook_callback = ReplaceCodebookCallback(model.get_vq_model(), batch_size=args.bs, frequency=args.replace_codebook)
    wandbImage = WandbImageCallback(model, val_dataset, log_freq=10)

    if args.test_run:
        callbacks = [reduce_lr, WandbCallback(save_model=False), wandbImage]
    else:
        if args.replace_codebook>0:
            callbacks = [reduce_lr, WandbCallback(save_model=False), wandbImage, model_checkpoint_callback, replace_codebook_callback]
        else: 
            callbacks = [reduce_lr, WandbCallback(save_model=False), wandbImage]

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
    history = model.fit(
        x=train_dataset,
        epochs=args.epochs,
        batch_size=args.bs,
        callbacks=callbacks,
        verbose=1,
        validation_data=val_dataset,
    )
    # Log the configuration and results
    # log_experiment_results(channels, num_embedding, embedding_dim, args, history)
    wandb.finish()


def log_experiment_results(channel_list, num_embedding, embedding_dim, args, history):
    log_file = f'/N/slate/aajais/experiment_logs/vqvae/{args.suffix}-log.csv'

    # Directly access metrics from history.history
    training_loss_list = history.history['loss']
    validation_loss_list = history.history['val_loss']
    training_q_loss_list = history.history['quantize_loss']
    validation_q_loss_list = history.history.get('val_quantize_loss', [None])
    training_recon_loss_list = history.history.get('reconst_loss', [None])  
    validation_recon_loss_list = history.history.get('val_reconst_loss', [None])
    perplexity_list = history.history.get('perplexity', [None])  
    ssim_list = history.history.get('ssim', [None])  
    psnr_list = history.history.get('psnr', [None])  
    
    # Prepare data for logging  
    data = {
        'Channel List': str(channel_list),
        'Num Embedding': num_embedding,
        'Embedding Dim': embedding_dim,
        'Training Loss List': str(training_loss_list),
        'Validation Loss List': str(validation_loss_list),
        'Training Quantize Loss List': str(training_q_loss_list),
        'Validation Quantize Loss List': str(validation_q_loss_list),
        'Training Recon Loss List': str(training_recon_loss_list),
        'Validation Recon Loss List': str(validation_recon_loss_list),
        'Perplexity List': str(perplexity_list),
        'SSIM List': str(ssim_list),
        'PSNR List': str(psnr_list),
    }

    # Append data to the log file
    if not os.path.isfile(log_file):
        pd.DataFrame([data]).to_csv(log_file, mode='w', index=False)
    else:
        pd.DataFrame([data]).to_csv(log_file, mode='a', index=False, header=False)

    print(f"Logged experiment results to {log_file}")


def run(args):
    # Initialize a new wandb run
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
    args.num_gpus = strategy.num_replicas_in_sync
    print(f"Number of devices: {args.num_gpus}")
    gpus = tf.config.experimental.list_physical_devices("GPU")
    print(gpus)
    print(tf.config.list_logical_devices())
    args.lbs = args.lbs
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

    dataset_save_path = (f"/N/slate/aajais/skullstripping_datasets/training_data/with_mask_context_B12-KR-AUG-all-T/")
    # dataset_save_path = (f"/N/slate/aajais/skullstripping_datasets/training_data/with_mask_context_{args.suffix}/")
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
        if args.train_vq:
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
    dataset = dataset.take(3000)
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
        run_experiment(args, strategy, train_dataset, val_dataset)
    elif args.test_vq:
        print(f"Testing Scaled VQVAE monai with ckpt - {args.suffix}-{args.test_epoch}")
        with strategy.scope():
            channels = eval(args.channel_list)  # Convert string to tuple
            num_embedding = args.num_embedding
            embedding_dim = args.embedding_dim

            model = build_and_compile_model(channels, num_embedding, embedding_dim, args, strategy)
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

if __name__ == "__main__":
    print(tf.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replace_codebook", type=int,default=0,help="Replace Codebook Frequency"
    )
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
    parser.add_argument(
        "--exp_name",
        type=str,
        default="wandb",
    )
    parser.add_argument("--test_vq", default=False, action="store_true", help="testing flag - VQVAE")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--lbs", type=int, default=3, help="Batch size per gpu")
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
    parser.add_argument("--save_best_only", default=True, action="store_true")
    parser.add_argument("--vqvae_load_ckpt", type=str, default=None)
    parser.add_argument("--resume_ckpt", type=str)
    parser.add_argument("--test_run", default=False, action="store_true")
    parser.add_argument("--channel_list", type=str, help="List of channels in the format '(num1,num2,...)'")
    parser.add_argument("--num_embedding", type=int, help="Number of embeddings")
    parser.add_argument("--embedding_dim", type=int, help="Dimension of embeddings")
    args = parser.parse_args()

    run(args)
