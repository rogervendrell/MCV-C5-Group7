#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D /tmp # working directory
#SBATCH -t 0-00:05 # Runtime in D-HH:MM
#SBATCH -p mhigh,mlow # Partition to submit to
#SBATCH -q masterlow  # This way will only requeue of dcc partition
#SBATCH --mem 4096 # 4GB memory
#SBATCH --gres gpu:1 # Request of 1 gpu
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written
sleep 5
/ghome/share/example/deviceQuery
nvidia-smi
