#!/bin/bash

#SBATCH -J sc_mae
#SBATCH -p gpu_p
#SBATCH --qos gpu_long
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH --mem=150GB
#SBATCH --nice=10000

conda activate celldreamer
python -u train.py 