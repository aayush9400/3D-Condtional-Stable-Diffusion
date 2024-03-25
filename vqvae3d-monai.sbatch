#!/bin/bash

#SBATCH --account r00230
#SBATCH -n 2
#SBATCH --mem=96G
#SBATCH -t 2-00:00:00
#SBATCH -J VQBothManiLoss
#SBATCH -p gpu
#SBATCH -o vqvae3d-monai-B12-both2.output

##SBATCH -N 1
##SBATCH --mail-type=ALL
##SBATCH --mail-user=varbayan@iu.edu


#SBATCH --gres=gpu:v100:4
##SBATCH --gpus-per-node=1

module load deeplearning/.2.9.1
python main.py --train --lbs 3 --both_datasets --resume_ckpt 31
# python main.py --train --lb 3 --cc359 --resume_ckpt 30
##--num_gpus 4 --suffix B9 
