#!/bin/bash
#SBATCH -n 4                    # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH -D /tmp                 # working directory
#SBATCH -J trsf_1
#SBATCH -t 1-00:00              # Runtime in D-HH:MM
#SBATCH -p mlow                 # Partition to submit to
#SBATCH -q masterlow
#SBATCH --mem 32768             # 32GB memory
#SBATCH --gres gpu:1            # Request of 1 gpu
#SBATCH -o /ghome/group07/MCV-C5-Group7/Week3/5-attention/output/%x_%u_%j.out
#SBATCH -e /ghome/group07/MCV-C5-Group7/Week3/5-attention/output/%x_%u_%j.err

sleep 5
source /ghome/group07/miniconda3/etc/profile.d/conda.sh
conda activate c5

python /ghome/group07/MCV-C5-Group7/Week3/5-attention/main.py \
    --num-decoder-layers 1
