#!/bin/bash

#SBATCH --account r00230
#SBATCH -n 2
#SBATCH --mem=96G
#SBATCH -t 2-00:00:00
#SBATCH -J dm3d5B8
#SBATCH -p gpu
#SBATCH -o dm3d-B8-DM500-both.output

##SBATCH -N 1
##SBATCH --mail-type=ALL
##SBATCH --mail-user=varbayan@iu.edu

#SBATCH --gres=gpu:v100:4
##SBATCH --gpus-per-node=2

# module unload deeplearning
# module load deeplearning/2.11.0
module load deeplearning/.2.9.1

# python main.py --train_dm --lbs 1 --vqvae_load_ckpt /N/slate/varbayan/checkpoints-vqvae-monai/B5/131.ckpt --timesteps 500

# python main.py --train_dm --lbs 1 --cc359 --vqvae_load_ckpt /N/slate/varbayan/checkpoints-vqvae-monai/CC-B10/44.ckpt --timesteps 500

python main.py --train_dm --lbs 2 --both_datasets --vqvae_load_ckpt /N/slate/varbayan/checkpoints-vqvae-monai/B12-both/69.ckpt --timesteps 500