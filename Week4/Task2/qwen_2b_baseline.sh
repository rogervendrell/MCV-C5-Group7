#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /tmp
#SBATCH -J qwen_2b_baseline
#SBATCH -t 0-06:00
#SBATCH -p mlow
#SBATCH -q masterlow
#SBATCH --mem 32768
#SBATCH --gres gpu:1
#SBATCH -o /ghome/group07/MCV-C5-Group7/Week4/Task2/output/%x_%u_%j.out
#SBATCH -e /ghome/group07/MCV-C5-Group7/Week4/Task2/output/%x_%u_%j.err

# Zero-shot baseline for Qwen3-VL-2B-Instruct (comparable to Llama 3.2-1B).

sleep 5
source /ghome/group07/miniconda3/etc/profile.d/conda.sh
conda activate c5clean

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

mkdir -p /ghome/group07/MCV-C5-Group7/Week4/Task2/output

python /ghome/group07/MCV-C5-Group7/Week4/Task2/main.py \
    --strategy baseline_2b
