#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /tmp
#SBATCH -J gen_vw
#SBATCH -t 2-00:00
#SBATCH -p mlow
#SBATCH -q masterlow
#SBATCH --mem 57344
#SBATCH --gres gpu:1
#SBATCH -o /ghome/group07/MCV-C5-Group7/Week5/task_d/output/%x_%u_%j.out
#SBATCH -e /ghome/group07/MCV-C5-Group7/Week5/task_d/output/%x_%u_%j.err

sleep 5
source /ghome/group07/miniconda3/etc/profile.d/conda.sh
conda activate c5clean

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# ── Tunable parameters ──────────────────────────────────────────────────────
# Model: sd-turbo  |  sdxl-turbo  |  sdxl
MODEL="sdxl-turbo"

# Generation mode:
#   degradation  – wrap each caption in a degradation prompt (blur/dark/close_up/…)
#   caption_only – use the raw reference caption as-is (clean synthetic images)
MODE="caption_only"

# Images per pipeline call — reduce if you hit OOM
#   sd-turbo  @ 512px  →  8–16 safe on 24 GB
#   sdxl-turbo @ 512px →  4–8
#   sdxl       @ 1024px → 2
BATCH_SIZE=8

SEED=42

# Save annotations.json every N images (also saves at the end)
CHECKPOINT_EVERY=1000
# ────────────────────────────────────────────────────────────────────────────

python /ghome/group07/MCV-C5-Group7/Week5/task_d/generate_augmented_dataset.py \
    --model            ${MODEL} \
    --mode             ${MODE} \
    --batch_size       ${BATCH_SIZE} \
    --seed             ${SEED} \
    --checkpoint_every ${CHECKPOINT_EVERY} \
    --scores_json  /ghome/group07/MCV-C5-Group7/Week5/task_d/results/clip_scores.json \
    --output_dir   /ghome/group07/MCV-C5-Group7/Week5/task_d/augmented_dataset \
    --run_id       ${SLURM_JOB_ID:-local}
