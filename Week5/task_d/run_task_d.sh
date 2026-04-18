#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /tmp
#SBATCH -J augment_vw
#SBATCH -t 2-00:00
#SBATCH -p mhigh
#SBATCH -q masterhigh
#SBATCH --mem 57344
#SBATCH --gres gpu:1
#SBATCH -o /ghome/group07/MCV-C5-Group7/Week5/task_d/output/%x_%u_%j.out
#SBATCH -e /ghome/group07/MCV-C5-Group7/Week5/task_d/output/%x_%u_%j.err

sleep 5
source /ghome/group07/miniconda3/etc/profile.d/conda.sh
conda activate c5clean

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# ── Tunable parameters ──────────────────────────────────────────────────────
# How many top-quality images to augment
N_IMAGES=5

# Degradation styles to apply (space-separated, subset of:
#   blur  dark  close_up  overexposed  low_quality)
STYLES="blur dark close_up overexposed low_quality"

# txt2img model: sdxl  |  sdxl-turbo  |  sd-turbo
MODEL="sd-turbo"

# Random seed for reproducibility
SEED=42
# ────────────────────────────────────────────────────────────────────────────

# Output goes to results/<model>/<slurm_job_id>/
# Falls back to "local" when run outside SLURM
RUN_ID="${SLURM_JOB_ID:-local}"
OUTPUT_DIR="/ghome/group07/MCV-C5-Group7/Week5/task_d/results/${MODEL}/${RUN_ID}"

# Requires clip_scores.json produced by run_clip_score.sh
python /ghome/group07/MCV-C5-Group7/Week5/task_d/task_d.py \
    --n_images  ${N_IMAGES} \
    --styles    ${STYLES} \
    --model     ${MODEL} \
    --seed      ${SEED} \
    --scores_json /ghome/group07/MCV-C5-Group7/Week5/task_d/results/clip_scores.json \
    --output_dir  "${OUTPUT_DIR}"
