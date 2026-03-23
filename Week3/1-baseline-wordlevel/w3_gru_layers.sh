#!/bin/bash
#SBATCH -n 4                    # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH -D /tmp                 # working directory
#SBATCH -J gru_layers
#SBATCH --array=1-3             # One job per GRU layer count (1, 2, 3)
#SBATCH -t 1-00:00              # Runtime in D-HH:MM
#SBATCH -p mlow                 # Partition to submit to
#SBATCH -q masterlow
#SBATCH --mem 32768             # 32GB memory
#SBATCH --gres gpu:1            # Request of 1 gpu
#SBATCH -o /ghome/group07/MCV-C5-Group7/Week3/1-baseline-wordlevel/output/%x_%u_%A_%a.out
#SBATCH -e /ghome/group07/MCV-C5-Group7/Week3/1-baseline-wordlevel/%x_%u_%A_%a.err

sleep 5
source /ghome/group07/miniconda3/etc/profile.d/conda.sh
conda activate c5

python /ghome/group07/MCV-C5-Group7/Week3/1-baseline-wordlevel/main.py \
    --num-decoder-layers "$SLURM_ARRAY_TASK_ID"
