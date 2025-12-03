#!/bin/bash
#SBATCH -J PestSpace
#SBATCH --partition gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --output=/gpfs/helios/home/fred87/PestSpace/Output/slurm_%x.%j.out # STDOUT

# Load modules:

module load cuda/11.7.0
module load cudnn/8.2.0.53-11.3

#Run model script:

/gpfs/helios/home/fred87/.conda/envs/pestspace/bin/python /gpfs/helios/home/fred87/PestSpace/train.py