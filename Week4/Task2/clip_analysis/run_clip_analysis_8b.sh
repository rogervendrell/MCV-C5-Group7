#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /tmp
#SBATCH -J clip_analysis_8b
#SBATCH -t 0-02:00
#SBATCH -p mlow
#SBATCH -q masterlow
#SBATCH --mem 32768
#SBATCH --gres gpu:1
#SBATCH -o /ghome/group07/MCV-C5-Group7/Week4/Task2/clip_analysis/output/%x_%u_%j.out
#SBATCH -e /ghome/group07/MCV-C5-Group7/Week4/Task2/clip_analysis/output/%x_%u_%j.err

sleep 5
source /ghome/group07/miniconda3/etc/profile.d/conda.sh
conda activate c5clean

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

mkdir -p /ghome/group07/MCV-C5-Group7/Week4/Task2/clip_analysis/output

python /ghome/group07/MCV-C5-Group7/Week4/Task2/clip_analysis/clip_analysis.py \
    --predictions /ghome/group07/MCV-C5-Group7/Week4/Task2/results/qwen_zero_shot_8b/109562/predictions_val.json \
    --output-dir  /ghome/group07/MCV-C5-Group7/Week4/Task2/clip_analysis/8b_zeroshot_109562
