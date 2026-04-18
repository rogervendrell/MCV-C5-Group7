#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /tmp
#SBATCH -J vizwiz_gen
#SBATCH -t 2-00:00
#SBATCH -p mhigh
#SBATCH -q masterhigh
#SBATCH --mem 32768
#SBATCH --gres gpu:1
#SBATCH -o /ghome/group07/MCV-C5-Group7/Week5/task_c/output/%x_%u_%j.out
#SBATCH -e /ghome/group07/MCV-C5-Group7/Week5/task_c/output/%x_%u_%j.err

sleep 5
source /ghome/group07/miniconda3/etc/profile.d/conda.sh
conda activate c5clean

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Number of VizWiz images to process
N_IMAGES=5

# Models to use for generation (choose from: sd-2-1, sd-turbo, sdxl, sdxl-turbo, sd-3-5-medium)
MODELS="sd-turbo sdxl-turbo"

python /ghome/group07/MCV-C5-Group7/Week5/task_c/task_c.py \
    --n_images ${N_IMAGES} \
    --models ${MODELS} \
    --seed 42 \
    --output_dir /ghome/group07/MCV-C5-Group7/Week5/task_c/output
