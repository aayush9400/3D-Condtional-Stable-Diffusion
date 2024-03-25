#!/bin/bash

#SBATCH -J NSVQB12-256
#SBATCH -p gpu
#SBATCH --account r00230
#SBATCH -o nsvqvae3d-B12-AUG-all-KR-256.output
#SBATCH -e nsvqvae3d-B12-AUG-all-KR-256.err
#SBATCH -n 2
#SBATCH --mem=128G
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aajais@iu.edu
#SBATCH --gres=gpu:4

module load python/gpu/3.10.10 
# module load deeplearning/.2.9.1
# python main.py --train --dataset='all' --lbs 2 --epochs 60
# python main.py --train --augment --kernel_resize --save_best_only --dataset='all' --lbs 3 --epochs 60
python main2.py --train_vq --augment --kernel_resize --dataset='all-T' --lbs 3 --epochs 200
