#!/bin/bash

#SBATCH --account r00230
#SBATCH -n 2
#SBATCH --mem=96G
#SBATCH -t 1:00:00
#SBATCH -J monaiVQtest
#SBATCH -p gpu

##SBATCH -N 1
#SBATCH -o vqvae3d-monai-test-B5.output
##SBATCH --mail-type=ALL
##SBATCH --mail-user=varbayan@iu.edu


#SBATCH --gres=gpu:v100:2
##SBATCH --gpus-per-node=1

module load deeplearning/.2.9.1
#python main.py --test --lbs 5 --test_epoch 130

# python main.py --test --lbs 5 --test_epoch 43 --cc359 --kernel_resize
# python main.py --test --lbs 3 --test_epoch 80 --cc359

python main.py --test --lbs 6 --test_epoch 91 --both_datasets





##--num_gpus 4 --suffix B9 
