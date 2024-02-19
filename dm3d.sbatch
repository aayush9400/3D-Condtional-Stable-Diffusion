#!/bin/bash

#SBATCH -J dm3d_B8_500
#SBATCH -p gpu
#SBATCH --account r00230
#SBATCH -o dm3d-B8-500-all-17.output
#SBATCH -e dm3d-B8-500-all-17.err
#SBATCH -n 2
#SBATCH --mem=96G
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aajais@iu.edu
#SBATCH --gpus-per-node=4

module load python/gpu/3.10.10 

# python main.py --train_dm --lbs 1 --vqvae_load_ckpt /N/slate/varbayan/checkpoints-vqvae-monai/B5/131.ckpt --timesteps 500

# python main.py --train_dm --lbs 1 --cc359 --vqvae_load_ckpt /N/slate/varbayan/checkpoints-vqvae-monai/CC-B10/44.ckpt --timesteps 500

# python main.py --train_dm --lbs 2 --both_datasets --vqvae_load_ckpt /N/slate/varbayan/checkpoints-vqvae-monai/B12-both/69.ckpt --timesteps 500
python main.py --train_dm --lbs 3 --dataset='all' --timesteps 500 --resume_ckpt 48 --epochs 100 --vqvae_load_ckpt /N/slate/aajais/vqvae_model/49.ckpt --save_best_only
