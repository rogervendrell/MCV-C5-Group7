import os
import csv
from datetime import datetime
from ultralytics import YOLO

# Slurm variables
job_name = os.environ.get("SLURM_JOB_NAME", "nojname")
user = os.environ.get("USER", "nouser")
job_id = os.environ.get("SLURM_JOB_ID", "nojobid")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

PROJECT_NAME = "/ghome/group07/MCV-C5-Group7/ultralytics/validate/runs"
run_name = "validate_final"

# best lr
# model = YOLO("/ghome/group07/MCV-C5-Group7/ultralytics/output/task_e/runs/detect/task_e_experiment/lr0.0001_augFalse/weights/best.pt")
# with data aug
# model = YOLO("/ghome/group07/MCV-C5-Group7/ultralytics/output/task_e/runs/detect/task_e_experiment/lr0.0001_augTrue/weights/best.pt")
# batch size 32
# model = YOLO("/ghome/group07/MCV-C5-Group7/ultralytics/output/task_e/runs/detect/task_e_bs/lr0.0001_augTrue/weights/best.pt")
# image size 960
# model = YOLO("/ghome/group07/MCV-C5-Group7/ultralytics/output/task_e/runs/detect/task_e_imgsz/lr0.0001_augTrue/weights/best.pt")
# image size 420
# model = YOLO("/ghome/group07/MCV-C5-Group7/ultralytics/output/task_e/runs/detect/task_e_imgsz_420/lr0.0001_augTrue/weights/best.pt")
# final training
model = YOLO("/ghome/group07/MCV-C5-Group7/ultralytics/output/task_e/runs/detect/task_e_final/lr0.0001_augTrue/weights/best.pt")


results = model.val(
    data="/ghome/group07/MCV-C5-Group7/ultralytics/kitti.yaml",
    imgsz=640,
    batch=16,
    project=PROJECT_NAME,   # main folder
    name=run_name,          # subfolder name
    workers=4,
    plots=True,
    # augmentation
)

# # Extract final metrics
# metrics = results.results_dict

# map50 = metrics.get("metrics/mAP50(B)", None)
# map5095 = metrics.get("metrics/mAP50-95(B)", None)
# precision = metrics.get("metrics/precision(B)", None)
# recall = metrics.get("metrics/recall(B)", None)