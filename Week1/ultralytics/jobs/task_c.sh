#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D /tmp # working directory
#SBATCH -t 0-02:00 # Runtime in D-HH:MM
#SBATCH -p mhigh # Partition to submit to
#SBATCH -q masterhigh # Required to requeue other users mlow queue jobs
                      # With this parameter only 1 job will be running in queue mhigh
                      # By defaulf the value is masterlow if not defined
#SBATCH --mem 16384 # 16GB memory
#SBATCH --gres gpu:1 # Request of 1 gpu
#SBATCH -o /ghome/group07/MCV-C5-Group7/ultralytics/output/task_c/%x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e /ghome/group07/MCV-C5-Group7/ultralytics/output/task_c/%x_%u_%j.err # File to which STDERR will be written
sleep 5
source /ghome/group07/miniconda3/etc/profile.d/conda.sh
conda activate c5
cd /ghome/group07/MCV-C5-Group7/ultralytics/output/task_c
python /ghome/group07/MCV-C5-Group7/ultralytics/task_c/task_c.py