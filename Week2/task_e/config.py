import torch

DATASET_ROOT = "/home/group07/mcv/datasets/C5/KITTI-MOTS"

SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"

MODEL_TYPE = "vit_b"

BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"