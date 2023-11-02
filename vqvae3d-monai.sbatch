#!/bin/bash

#SBATCH --account r00230
#SBATCH -n 2
#SBATCH --mem=96G
#SBATCH -t 2-00:00:00
#SBATCH -J VQBothScaledLoss
#SBATCH -p gpu
#SBATCH -o vqvae3d-scaled-monai-B16-all.output
#SBATCH --gres=gpu:v100:4

module load deeplearning/.2.9.1
python main.py --train --lbs 8 --dataset='all' --epochs 60
# python main.py --train --lbs 2 --dataset='both' --resume_ckpt 33 --epochs 60 
##--num_gpus 4 --suffix B9 
