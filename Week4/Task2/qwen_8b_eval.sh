#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /tmp
#SBATCH -J qwen_8b_eval
#SBATCH -t 1-00:00
#SBATCH -p mhigh
#SBATCH -q masterhigh
#SBATCH --mem 57344
#SBATCH --gres gpu:1
#SBATCH -o /ghome/group07/MCV-C5-Group7/Week4/Task2/output/%x_%u_%j.out
#SBATCH -e /ghome/group07/MCV-C5-Group7/Week4/Task2/output/%x_%u_%j.err

# Zero-shot evaluation with Qwen3-VL-8B-Instruct (comparable to Llama 3.2-11B).
# No training; the model is queried out of the box on the VizWiz validation set.
# Requires mhigh because the 8B model in bfloat16 needs ~16 GB GPU memory.

sleep 5
source /ghome/group07/miniconda3/etc/profile.d/conda.sh
conda activate c5clean

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

mkdir -p /ghome/group07/MCV-C5-Group7/Week4/Task2/output

python /ghome/group07/MCV-C5-Group7/Week4/Task2/main.py \
    --strategy zero_shot_8b
