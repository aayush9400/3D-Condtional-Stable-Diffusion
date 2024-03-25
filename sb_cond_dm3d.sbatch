#!/bin/bash

#SBATCH -J conditional_dm3d_B12_500
#SBATCH -p gpu
#SBATCH --account r00230
#SBATCH -o conditional_dm3d_B12_500.output
#SBATCH -e conditional_dm3d_B12_500.err
#SBATCH -n 2
#SBATCH --mem=96G
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aajais@iu.edu
#SBATCH --gpus-per-node=4

module load python/gpu/3.10.10 

python main_conditional_dm.py --train_dm --lbs 3 --dataset='all-T' --timesteps 500 --epochs 100 --vqvae_load_ckpt /N/slate/aajais/checkpoints-vqvae-monai-scaled-128/B12-KR-AUG-all-T/78.ckpt --augment --kernel_resize --save_best_only