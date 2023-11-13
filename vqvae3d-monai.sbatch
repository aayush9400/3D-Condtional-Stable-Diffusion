#!/bin/bash

#SBATCH --account r00230
#SBATCH -n 2
#SBATCH --mem=96G
#SBATCH -t 2-00:00:00
#SBATCH -J VQAllScaledLoss
#SBATCH -p gpu
#SBATCH -o vqvae3d-scaled-monai-B12-all-KR-QL.output
#SBATCH --gres=gpu:v100:4

module load deeplearning/.2.9.1
# python main.py --train --dataset='all' --lbs 2 --epochs 60
python main.py --train --kernel_resize --save_best_only --dataset='all' --lbs 3 --epochs 60
# python main.py --train  --kernel_resize --lbs 2 --dataset='all' --resume_ckpt 24 --epochs 60  
##--num_gpus 4 --suffix B9 
