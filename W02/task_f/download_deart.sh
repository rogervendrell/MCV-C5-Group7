#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /tmp
#SBATCH -J deart_download
#SBATCH -t 1-00:00
#SBATCH -p mhigh
#SBATCH -q masterhigh
#SBATCH --mem 32768
#SBATCH --gres gpu:1
#SBATCH -o /ghome/group07/MCV-C5-Group7/Week2/task_f/output/%x_%u_%j.out
#SBATCH -e /ghome/group07/MCV-C5-Group7/Week2/task_f/output/%x_%u_%j.err
sleep 5
source /ghome/group07/miniconda3/etc/profile.d/conda.sh
conda activate c5
cd /ghome/group07/MCV-C5-Group7/Week2/task_f
python download_deart.py