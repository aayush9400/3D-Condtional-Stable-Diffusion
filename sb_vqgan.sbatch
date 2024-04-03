#!/bin/bash

#SBATCH -J LPIPS-VQGB12
#SBATCH -p gpu
#SBATCH --account r00230
#SBATCH -o vqgan-128-512x256-12.output
#SBATCH -e vqgan-128-512x256-12.err
#SBATCH -n 2
#SBATCH --mem=96G
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aajais@iu.edu
#SBATCH --gres=gpu:4

module load python/gpu/3.10.10 
python main_exp_vqgan.py --channel_list='(32,64,128)' --dataset='all-T' --exp_name='12,32x64x128,512x256' --lbs=3 --lr=0.0001 --num_embedding=512 --embedding_dim=256 --epochs=400 
# python main_exp_vqgan.py --channel_list='(32,64,128,256)' --dataset='all-T' --exp_name='12,32x64x128x256,512x256' --lbs=3 --lr=0.0001 --num_embedding=512 --embedding_dim=256 --dropout=0.2 --epochs=400 