#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /tmp
#SBATCH -J clip_score
#SBATCH -t 2-00:00
#SBATCH -p mhigh
#SBATCH -q masterhigh
#SBATCH --mem 32768
#SBATCH --gres gpu:1
#SBATCH -o /ghome/group07/MCV-C5-Group7/Week5/task_d/output/%x_%u_%j.out
#SBATCH -e /ghome/group07/MCV-C5-Group7/Week5/task_d/output/%x_%u_%j.err

sleep 5
source /ghome/group07/miniconda3/etc/profile.d/conda.sh
conda activate c5clean

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Images are batched through CLIP; increase if you have >=24 GB VRAM
BATCH_SIZE=64

python /ghome/group07/MCV-C5-Group7/Week5/task_d/clip_score.py \
    --batch_size ${BATCH_SIZE} \
    --output_json /ghome/group07/MCV-C5-Group7/Week5/task_d/results/clip_scores.json
