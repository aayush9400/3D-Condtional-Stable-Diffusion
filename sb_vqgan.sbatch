#!/bin/bash

#SBATCH -J VQB24
#SBATCH -p gpu
#SBATCH --account r00230
#SBATCH -o vqgan-512x256-12.output
#SBATCH -e vqgan-512x256-12.err
#SBATCH -n 2
#SBATCH --mem=96G
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aajais@iu.edu
#SBATCH --gres=gpu:4

module load python/gpu/3.10.10 
python main_exp_vqgan.py --channel_list='(32,64,128)' --dataset='all-T' --exp_name='12,32x64x128,512x256' --embedding_dim=256 --epochs=200  --lbs=3 --lr=0.0001 --num_embedding=512