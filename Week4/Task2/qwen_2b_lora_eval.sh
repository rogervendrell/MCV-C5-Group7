#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /tmp
#SBATCH -J qwen_2b_lora_eval
#SBATCH -t 0-04:00
#SBATCH -p mlow
#SBATCH -q masterlow
#SBATCH --mem 48000
#SBATCH --gres gpu:1
#SBATCH -o /ghome/group07/MCV-C5-Group7/Week4/Task2/output/%x_%u_%j.out
#SBATCH -e /ghome/group07/MCV-C5-Group7/Week4/Task2/output/%x_%u_%j.err

# Eval-only pass for lora_2b: loads best_model.pt and runs one validation pass.
# Saves summary.json and predictions_val.json without any training.

VIT_CHECKPOINT=/ghome/group07/MCV-C5-Group7/Week4/Task1_ft/results/vit_gpt2_vit_ft_gpt_frozen_enc4L_frozen/109481/weights/best_model.pt

sleep 5
source /ghome/group07/miniconda3/etc/profile.d/conda.sh
conda activate c5clean

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /ghome/group07/MCV-C5-Group7/Week4/Task2/output

python /ghome/group07/MCV-C5-Group7/Week4/Task2/main.py \
    --strategy       lora_2b \
    --vit-checkpoint "$VIT_CHECKPOINT" \
    --eval-only
