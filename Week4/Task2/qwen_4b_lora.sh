#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /tmp
#SBATCH -J qwen_4b_lora
#SBATCH -t 2-00:00
#SBATCH -p mhigh
#SBATCH -q masterhigh
#SBATCH --mem 57344
#SBATCH --gres gpu:1
#SBATCH -o /ghome/group07/MCV-C5-Group7/Week4/Task2/output/%x_%u_%j.out
#SBATCH -e /ghome/group07/MCV-C5-Group7/Week4/Task2/output/%x_%u_%j.err

# LoRA fine-tuning: frozen ViT (Task1_ft best weights) + Qwen3-4B decoder.
# Architecture: FrozenViT → linear projection → LoRA Qwen3-4B (text-only).
# The ViT is the best Task1_ft checkpoint (vit_ft_gpt_frozen, enc last 4L, 30 epochs).
# Uses mhigh because the 4B model needs more GPU memory than 2B.
# Compare results against qwen_4b_baseline.sh (Qwen3-VL-4B zero-shot baseline).

VIT_CHECKPOINT=/ghome/group07/MCV-C5-Group7/Week4/Task1_ft/results/vit_gpt2_vit_ft_gpt_frozen_enc4L_frozen/109481/weights/best_model.pt

sleep 5
source /ghome/group07/miniconda3/etc/profile.d/conda.sh
conda activate c5clean

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /ghome/group07/MCV-C5-Group7/Week4/Task2/output

python /ghome/group07/MCV-C5-Group7/Week4/Task2/main.py \
    --strategy       lora_4b \
    --vit-checkpoint "$VIT_CHECKPOINT"
