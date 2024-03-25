#!/bin/bash

#SBATCH --account r00230
#SBATCH -n 2
#SBATCH --mem=96G
#SBATCH -t 1-00:00:00
#SBATCH -J dm3d
#SBATCH -p gpu
#SBATCH -o dm3d-test.output

##SBATCH -N 1
##SBATCH --mail-type=ALL
##SBATCH --mail-user=varbayan@iu.edu


#SBATCH --gres=gpu:v100:1

module load deeplearning/.2.9.1

# python main.py --test_dm --lbs 10 --test_epoch 109 --vqvae_load_ckpt /N/slate/varbayan/checkpoints-vqvae-monai/B5/130.ckpt
# python main.py --test_dm --lbs 8 --test_epoch 146 --vqvae_load_ckpt /N/slate/varbayan/checkpoints-vqvae-monai/B5/130.ckpt --timesteps 1000

# python main.py --test_dm --lbs 4 --cc359 --timesteps 1000 --test_epoch 86 --vqvae_load_ckpt /N/slate/varbayan/checkpoints-vqvae-monai/CC-B10/43.ckpt
# python main.py --test_dm --lbs 4 --cc359 --timesteps 500 --test_epoch 42 --vqvae_load_ckpt /N/slate/varbayan/checkpoints-vqvae-monai/CC-B10/43.ckpt
# python main.py --test_dm --lbs 4 --cc359 --timesteps 500 --test_epoch 6 --vqvae_load_ckpt /N/slate/varbayan/checkpoints-vqvae-monai/CC-B10/43.ckpt

python main.py --test_dm --lbs 4 --both_datasets --timesteps 300 --test_epoch 51 --vqvae_load_ckpt /N/slate/varbayan/checkpoints-vqvae-monai/B12-both/69.ckpt
python main.py --test_dm --lbs 4 --both_datasets --timesteps 500 --test_epoch 58 --vqvae_load_ckpt /N/slate/varbayan/checkpoints-vqvae-monai/B12-both/69.ckpt
python main.py --test_dm --lbs 8 --both_datasets --timesteps 500 --test_epoch 57 --vqvae_load_ckpt /N/slate/varbayan/checkpoints-vqvae-monai/B12-both/69.ckpt
