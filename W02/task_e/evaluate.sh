#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D /tmp # working directory
#SBATCH -J inf_e2
#SBATCH -t 1-00:00 # Runtime in D-HH:MM
#SBATCH -p mhigh # Partition to submit to
#SBATCH -q masterlow 
#SBATCH --mem 32768 # 32GB memory
#SBATCH --gres gpu:1 # Request of 1 gpu
#SBATCH -o /ghome/group07/MCV-C5-Group7/Week2/task_e/output/%x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e /ghome/group07/MCV-C5-Group7/Week2/task_e/output/%x_%u_%j.err # File to which STDERR will be written
sleep 5
source /ghome/group07/miniconda3/etc/profile.d/conda.sh
conda activate c5
cd /ghome/group07/MCV-C5-Group7/Week2/task_e/
python /ghome/group07/MCV-C5-Group7/Week2/task_e/evaluate_sam.py