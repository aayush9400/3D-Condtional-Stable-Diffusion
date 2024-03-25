#!/bin/bash

#SBATCH -J VQB24
#SBATCH -p gpu
#SBATCH --account r00230
#SBATCH -o vqvae3d-512x256-12.output
#SBATCH -e vqvae3d-512x256-12.err
#SBATCH -n 2
#SBATCH --mem=96G
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aajais@iu.edu
#SBATCH --gres=gpu:4

module load python/gpu/3.10.10 
# module load deeplearning/.2.9.1
# python main.py --train --dataset='all' --lbs 2 --epochs 60
# python main.py --train --augment --kernel_resize --save_best_only --dataset='all' --lbs 3 --epochs 60
# python main.py --train_vq  --augment --kernel_resize --save_best_only --dataset='all-T' --lbs 3 --lr 0.0001 --epochs 200 #--resume_ckpt 3
python exp_main.py --channel_list='(32,64,128)' --dataset='all-T' --exp_name='24,32x64x128,512x256' --embedding_dim=256 --epochs=200  --lbs=3 --lr=0.0001 --num_embedding=512 --replace_codebook=100