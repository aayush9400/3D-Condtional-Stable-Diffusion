#!/bin/bash

#SBATCH -J nsvq_sweep
#SBATCH -p gpu
#SBATCH --account r00230
#SBATCH -o nsvq_sweep.output
#SBATCH -e nsvq_sweep.err
#SBATCH -n 2
#SBATCH --mem=128G
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aajais@iu.edu
#SBATCH --gpus-per-node=4

module load python/gpu/3.10.10 

export WANDB_PROJECT=nsvqvae
export WANDB_ENTITY=dipy_genai

# wandb agent jaayush12/nsvqvae/195l0r1r # rpl_code=5
# wandb agent jaayush12/nsvqvae/8077wx92 # rpl_code=15

# new sweep
wandb agent dipy_genai/nsvqvae/ievqma73 # rpl_code=15
