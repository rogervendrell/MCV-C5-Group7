#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -p mlow
#SBATCH -q masterlow
#SBATCH --mem=16384
#SBATCH --gres=gpu:1
#SBATCH --job-name=val_domshft
#SBATCH -o %x_%u_%j.out
#SBATCH -e %x_%u_%j.err

RUN_DIR=/ghome/group07/MCV-C5-Group7/domainshift/validate/runs/${SLURM_JOB_NAME}_${SLURM_JOB_USER}_${SLURM_JOB_ID}/

mkdir -p "$RUN_DIR"
cd "$RUN_DIR" || exit 1

sleep 5

# Load conda environment and run the script
source ~/miniconda3/etc/profile.d/conda.sh
conda activate c5
python -u /ghome/group07/MCV-C5-Group7/domainshift/validate_models.py