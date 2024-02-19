#!/bin/bash

#SBATCH -J VQB12T_2.9.1
#SBATCH -p gpu
#SBATCH --account r00230
#SBATCH -o vqvae3d-scaled-monai-B12-AUG-all-KR-18-CD.output
#SBATCH -e vqvae3d-scaled-monai-B12-AUG-all-KR-18-CD.err
#SBATCH -n 2
#SBATCH --mem=96G
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aajais@iu.edu
#SBATCH --gpus-per-node=4

# module load python/gpu/3.10.10 
module load deeplearning/.2.9.1
# python main.py --train --dataset='all' --lbs 2 --epochs 60
# python main.py --train --augment --kernel_resize --save_best_only --dataset='all' --lbs 3 --epochs 60
python main.py --train_vq  --augment --kernel_resize --save_best_only --dataset='all-T' --lbs 3 --lr 0.00001 --epochs 200 --create_dataset #--resume_ckpt 3