import time
import argparse
import os
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras

from dataset_utils import create_dataset, load_dataset, get_dataset_list
from networks.vqgan_gnorm import VQGAN, WandbImageCallback, ReplaceCodebookCallback, EpochCounterCallback

import wandb
from wandb.keras import WandbCallback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1 = INFO, 2 = WARNING, 3 = ERROR)
tf.get_logger().setLevel('ERROR')


def build_and_compile_model(channel_list, num_embedding, embedding_dim, args, strategy):
    with strategy.scope():
        # Dynamically set the number of residual layers or any other parameter based on channel_list
        num_res_layers = len(channel_list)  
        
        # Dynamically construct downsample and upsample parameters
        downsample_parameters = [(2, 4, 1, "same") for _ in channel_list]  
        upsample_parameters = [(2, 4, 1, "same", 0) for _ in channel_list]  

        model = VQGAN(
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
            dropout=args.dropout,
            B=args.bs,
            disc_threshold=args.disc_threshold,
            disc_loss_fn=args.disc_loss_fn,
            disc_use_sigmoid=args.disc_use_sigmoid,
            lpips_wt=args.lpips_wt,
            gan_feat_wt=args.gan_feat_wt,
            act_fn=args.act_fn,
        )

        x = tf.keras.layers.Input(shape=(128, 128, 128, 2))
        dummy_input_3d = tf.random.normal((1, 128, 128, 128, 1))
        _ = model.discriminator(dummy_input_3d)
        dummy_input_2d = tf.random.normal((1, 128, 128, 1))
        _ = model.discriminator_2d(dummy_input_2d)

        m = tf.keras.Model(inputs=[x], outputs=model(x))
        print(m.summary())
        model.compile(vqvae_optimizer=keras.optimizers.Adam(learning_rate=args.lr),
                      discriminator_optimizer=keras.optimizers.Adam(learning_rate=args.lr))

    return model


def run_experiment(args, strategy, train_dataset, val_dataset):
    run_name = f"{args.channel_list}|{args.num_embedding}x{args.embedding_dim}|{args.disc_threshold}|{args.disc_loss_fn}|{args.dataset}|gnorm" 
    wandb.init(project='vqgan', entity='dipy_genai', name=run_name, config=args, dir='/N/slate/aajais/')
    # Use wandb.config to access the hyperparameters
    channels = eval(args.channel_list)  # Convert string to tuple
    num_embedding = args.num_embedding
    embedding_dim = args.embedding_dim

    model = build_and_compile_model(channels, num_embedding, embedding_dim, args, strategy)
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"/N/slate/aajais/checkpoints-vqgan-monai-scaled-128/{args.suffix}/"
        + "{epoch}.ckpt",
        save_weights_only=True,
        save_best_only=args.save_best_only,
    )

    csv_logger = tf.keras.callbacks.CSVLogger(
        f"/N/slate/aajais/checkpoints-vqgan-monai-scaled-128/{args.suffix}/training.log",
        append=True,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss",
        factor=0.02,
        patience=8,
        min_lr=1e-5,
        verbose=1,
    )
    
    replace_codebook_callback = ReplaceCodebookCallback(model.get_vq_model(), batch_size=args.bs, frequency=args.replace_codebook)
    wandbImage = WandbImageCallback(model, val_dataset, log_freq=10)
    epoch_counter = EpochCounterCallback(model)
    if args.test_run:
        # callbacks = [reduce_lr, replace_codebook_callback, model_checkpoint_callback]
        callbacks = [reduce_lr, WandbCallback(save_model=False), wandbImage, epoch_counter]
    else:
        if args.replace_codebook>0:
            callbacks = [reduce_lr, WandbCallback(save_model=False), wandbImage, replace_codebook_callback, epoch_counter]
        else: 
            callbacks = [WandbCallback(save_model=False), wandbImage, epoch_counter, reduce_lr]

    initial_epoch = 0
    if args.resume_ckpt:
        model.load_weights(
            f"/N/slate/aajais/checkpoints-vqgan-monai-scaled-128/{args.suffix}/"
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
        callbacks=callbacks,
        verbose=1,
        validation_data=val_dataset,
    )
    # Log the configuration and results
    wandb.finish()


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

    dataset_list = get_dataset_list(dataset_vers=args.dataset, test_run_flag=args.test_run)
    print("Total Images in dataset: ", len(dataset_list))
    args.test_size = len(dataset_list) - (len(dataset_list) // args.bs) * args.bs
    print(args)

    lis = dataset_list[: -args.test_size] if args.test_size > 0 else dataset_list[:]
    if args.test_vq:
        lis = dataset_list[-args.test_size :]
    print("Total images available for training: ", len(lis))
    
    if args.dataset == "all-T":
        dataset_save_path = (f"/N/slate/aajais/skullstripping_datasets/training_data/with_mask_context_B12-KR-AUG-all-T/")
    elif args.dataset == "minus121":
        dataset_save_path = (f"/N/slate/aajais/skullstripping_datasets/training_data/minus121/")
    elif args.dataset == "minus121_augment":
        dataset_save_path = (f"/N/slate/aajais/skullstripping_datasets/training_data/minus121_augment/")
    # dataset_save_path = (f"/N/slate/aajais/skullstripping_datasets/training_data/with_mask_context_B12-KR-AUG-all/")
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
    if args.test_run:
        dataset = dataset.take(3 * args.bs)
    else:
        if args.dataset == 'all-T':
            dataset = dataset.take(3000)
        elif args.dataset == 'minus121':
            dataset = dataset
        
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

        train_dataset = train_dataset.shuffle(buffer_size = train_dataset.cardinality(), seed=42) 
        train_dataset = train_dataset.batch(args.bs).prefetch(
            tf.data.experimental.AUTOTUNE
        )

        # val_dataset = val_dataset.shuffle(buffer_size = val_dataset.cardinality(), seed=42) 
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

        print(f"Training Scaled vqgan monai")
        run_experiment(args, strategy, train_dataset, val_dataset)
    elif args.test_vq:
        print(f"Testing Scaled vqgan monai with ckpt - {args.suffix}-{args.test_epoch}")
        with strategy.scope():
            channels = eval(args.channel_list)  # Convert string to tuple
            num_embedding = args.num_embedding
            embedding_dim = args.embedding_dim

            model = build_and_compile_model(channels, num_embedding, embedding_dim, args, strategy)
        model.load_weights(
            os.path.join(
                "/N/slate/aajais/checkpoints-vqgan-monai-scaled-128",
                args.suffix,
                str(args.test_epoch) + ".ckpt",
            )
        )
        test_dataset = dataset.batch(args.bs).prefetch(tf.data.experimental.AUTOTUNE)

        directory = f"/N/slate/aajais/reconst-vqgan-monai-scaled-128/{args.suffix}/"
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
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace_codebook", type=int,default=0,help="Replace Codebook Frequency")
    parser.add_argument("--create_dataset", default=False, action="store_true", help="Create Dataset")
    parser.add_argument("--augment", default=False, action="store_true", help="Augment Data (F,B,C)")
    parser.add_argument("--train_vq", default=True, action="store_true", help="training flag - vqgan")
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
    parser.add_argument("--test_vq", default=False, action="store_true", help="testing flag - vqgan")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--lbs", type=int, default=3, help="Batch size per gpu")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs")
    parser.add_argument("--val_perc", type=float, default=0.1, help="Validation Percentage of Dataset")
    parser.add_argument(
        "--suffix",
        default="wandb",
        type=str,
        help="output or ckpts saved with this suffix",
    )
    parser.add_argument("--num_gpus", default=2, type=int, help="Number of GPUs to be used")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout value")
    parser.add_argument("--kernel_resize", default=False, action="store_true", help="kernel resize flag")
    parser.add_argument("--test_epoch", type=int)
    parser.add_argument("--save_best_only", default=True, action="store_true")
    parser.add_argument("--vqgan_load_ckpt", type=str, default=None)
    parser.add_argument("--resume_ckpt", type=str)
    parser.add_argument("--test_run", default=False, action="store_true")
    parser.add_argument("--channel_list", type=str, help="List of channels in the format '(num1,num2,...)'")
    parser.add_argument("--num_embedding", type=int, help="Number of embeddings")
    parser.add_argument("--embedding_dim", type=int, help="Dimension of embeddings")
    parser.add_argument("--disc_threshold", type=int, default=0, help="Training Steps to start Discriminator training")
    parser.add_argument("--disc_loss_fn", type=str, default='vanilla', help="Loss function to be used to calculate discriminator loss (vanilla, hinge)")
    parser.add_argument("--gan_feat_wt", default=4.0, type=float, help="Dropout value")
    parser.add_argument("--lpips_wt", default=4.0, type=float, help="Dropout value")    
    parser.add_argument("--disc_use_sigmoid", type=str2bool, default=False, help="Enable or disable sigmoid in discriminator")
    parser.add_argument("--act_fn", type=str, default='prelu', help="Activation to be use for Encode and Decoder")
    args = parser.parse_args()

    run(args)
