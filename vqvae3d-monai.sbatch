#!/bin/bash

#SBATCH --account r00230
#SBATCH -n 2
#SBATCH --mem=96G
#SBATCH -t 2-00:00:00
#SBATCH -J VQAllScaledLoss
#SBATCH -p gpu
#SBATCH -o vqvae3d-scaled-monai-B12-AUG-all-KR-Rls.output
#SBATCH --gres=gpu:v100:4

module load deeplearning/.2.9.1
# python main.py --train --dataset='all' --lbs 2 --epochs 60
# python main.py --train --augment --kernel_resize --save_best_only --dataset='all' --lbs 3 --epochs 60
python main.py --train  --augment --kernel_resize --dataset='all' --lbs 3 --lr 0.00001 --epochs 60 --resume_ckpt 37 
##--num_gpus 4 --suffix B9 
