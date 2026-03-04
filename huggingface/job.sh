#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -p mhigh
#SBATCH -q masterlow
#SBATCH --mem=16384
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_d
#SBATCH -o %x_%u_%j.out
#SBATCH -e %x_%u_%j.err

RUN_DIR=/ghome/group07/MCV-C5-Group7/huggingface/runs/${SLURM_JOB_NAME}_${SLURM_JOB_USER}_${SLURM_JOB_ID}/

mkdir -p "$RUN_DIR"
cd "$RUN_DIR" || exit 1

sleep 5

# Load conda environment and run the script
source ~/miniconda3/etc/profile.d/conda.sh
conda activate c3
python /ghome/group07/MCV-C5-Group7/huggingface/task_e/task_e.py