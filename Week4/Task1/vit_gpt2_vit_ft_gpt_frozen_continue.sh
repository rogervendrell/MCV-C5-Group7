#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /tmp
#SBATCH -J ft_vit_ft_gpt_frozen_cont
#SBATCH -t 2-00:00
#SBATCH -p mlow
#SBATCH -q masterlow
#SBATCH --mem 32768
#SBATCH --gres gpu:1
#SBATCH -o /ghome/group07/MCV-C5-Group7/Week4/Task1_ft/output/%x_%u_%j.out
#SBATCH -e /ghome/group07/MCV-C5-Group7/Week4/Task1_ft/output/%x_%u_%j.err

sleep 5
source /ghome/group07/miniconda3/etc/profile.d/conda.sh
conda activate c5clean

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Continue training vit_ft_gpt_frozen (enc last 4 layers) for 20 more epochs,
# resuming from the best checkpoint of the previous 10-epoch run (job 109280).
# All other hyperparameters are kept identical to the original run.
# Results are saved to a new job-ID subdirectory under the same experiment folder,
# so the original run's artifacts are not overwritten.
python /ghome/group07/MCV-C5-Group7/Week4/Task1_ft/main.py \
    --strategy          vit_ft_gpt_frozen \
    --encoder-mode      last_n \
    --encoder-ft-layers 4 \
    --epochs            20 \
    --resume-from       /ghome/group07/MCV-C5-Group7/Week4/Task1_ft/results/vit_gpt2_vit_ft_gpt_frozen_enc4L_frozen/109280/weights/best_model.pt \
    --lr                1e-5 \
    --weight-decay      0.01 \
    --label-smoothing   0.1 \
    --grad-clip         1.0 \
    --repetition-penalty 1.3
