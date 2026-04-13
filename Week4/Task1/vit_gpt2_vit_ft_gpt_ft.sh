#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /tmp
#SBATCH -J ft_vit_ft_gpt_ft
#SBATCH -t 1-00:00
#SBATCH -p mlow
#SBATCH -q masterlow
#SBATCH --mem 32768
#SBATCH --gres gpu:1
#SBATCH -o /ghome/group07/MCV-C5-Group7/Week4/Task1_ft/output/%x_%u_%j.out
#SBATCH -e /ghome/group07/MCV-C5-Group7/Week4/Task1_ft/output/%x_%u_%j.err

sleep 5
source /ghome/group07/miniconda3/etc/profile.d/conda.sh
conda activate c5clean

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Fine-tune last 4 ViT blocks + GPT-2 cross-attention layers.
python /ghome/group07/MCV-C5-Group7/Week4/Task1_ft/main.py \
    --strategy          vit_ft_gpt_ft \
    --encoder-mode      last_n \
    --encoder-ft-layers 4 \
    --decoder-mode      cross_attn \
    --lr                1e-5 \
    --weight-decay      0.01 \
    --label-smoothing   0.1 \
    --grad-clip         1.0 \
    --repetition-penalty 1.3
