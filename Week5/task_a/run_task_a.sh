#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /tmp
#SBATCH -J sd_compare
#SBATCH -t 2-00:00
#SBATCH -p mlow
#SBATCH -q masterlow
#SBATCH --mem 32768
#SBATCH --gres gpu:1
#SBATCH -o /ghome/group07/MCV-C5-Group7/Week5/task_a/output/%x_%u_%j.out
#SBATCH -e /ghome/group07/MCV-C5-Group7/Week5/task_a/output/%x_%u_%j.err

sleep 5
source /ghome/group07/miniconda3/etc/profile.d/conda.sh
conda activate c5clean

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

python /ghome/group07/MCV-C5-Group7/Week5/task_a/compare_models.py \
    --models sd-turbo sdxl sdxl-turbo sd-3-5-medium \
    --output_dir /ghome/group07/MCV-C5-Group7/Week5/task_a/output/out