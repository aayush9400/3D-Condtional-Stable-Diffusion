#!/bin/bash

#SBATCH -J vq_sweep
#SBATCH -p gpu
#SBATCH --account r00230
#SBATCH -o vq_sweep.output
#SBATCH -e vq_sweep.err
#SBATCH -n 2
#SBATCH --mem=96G
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aajais@iu.edu
#SBATCH --gpus-per-node=4

module load python/gpu/3.10.10 

# export WANDB_PROJECT=vqvae
# export WANDB_ENTITY=jaayush12
# export WANDB_ENTITY=dipy_genai
# wandb agent v6c46689
wandb agent dipy_genai/vqvae/isfh9x94