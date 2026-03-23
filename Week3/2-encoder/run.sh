#!/bin/bash
#SBATCH -n 4                      # Number of CPU cores
#SBATCH -N 1                      # All cores on one machine
#SBATCH -D /tmp                   # Working directory
#SBATCH -J uf_encoder               # Job name
#SBATCH -t 1-00:00                # Runtime D-HH:MM
#SBATCH -p mhigh                   # Partition
#SBATCH -q masterhigh              # QOS
#SBATCH --mem 32768               # 32 GB RAM
#SBATCH --gres gpu:1              # 1 GPU
#SBATCH -o /ghome/group07/MCV-C5-Group7/Week3/2-encoder/output/%x_%u_%j.out
#SBATCH -e /ghome/group07/MCV-C5-Group7/Week3/2-encoder/output/%x_%u_%j.err

sleep 5
source /ghome/group07/miniconda3/etc/profile.d/conda.sh
conda activate c5

python /ghome/group07/MCV-C5-Group7/Week3/2-encoder/main.py
