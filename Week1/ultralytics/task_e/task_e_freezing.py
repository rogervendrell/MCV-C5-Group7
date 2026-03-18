import os
import csv
from datetime import datetime
from ultralytics import YOLO

# Slurm variables
job_name = os.environ.get("SLURM_JOB_NAME", "nojname")
user = os.environ.get("USER", "nouser")
job_id = os.environ.get("SLURM_JOB_ID", "nojobid")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

FREEZE_STAGES = [22, 19, 10]

BATCH_SIZE = 32
WEIGHT_DECAY = 0.0005
OPTIMIZER = "AdamW"

PROJECT_NAME = "task_e_progressive_unfreeze"
RESULTS_CSV = f"/ghome/group07/MCV-C5-Group7/ultralytics/output/task_e/{job_name}_{user}_{job_id}_{timestamp}.csv"

DATA_YAML = "/ghome/group07/MCV-C5-Group7/ultralytics/kitti.yaml"
BASE_WEIGHTS = "/ghome/group07/MCV-C5-Group7/ultralytics/yolo26n.pt"

# CSV header
if not os.path.exists(RESULTS_CSV):
    with open(RESULTS_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "stage",
            "lr",
            "freeze_layers",
            "mAP50",
            "mAP50-95",
            "precision",
            "recall"
        ])

print("\n===== Starting Progressive Unfreezing Experiment =====\n")

for freeze in FREEZE_STAGES:

    stage_name = f"stage_{freeze}"
    lr = 0.0001
    epochs = 25

    print(f"\n===== {stage_name} | LR={lr} | Freeze={freeze} =====\n")

    model = YOLO(BASE_WEIGHTS)

    results = model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=640,
        batch=BATCH_SIZE,
        lr0=lr,
        optimizer=OPTIMIZER,
        weight_decay=WEIGHT_DECAY,
        project=PROJECT_NAME,
        name=stage_name,
        exist_ok=True,
        save=True,
        cache=True,
        workers=4,
        seed=42,
        val=True,
        plots=True,
        freeze=freeze,
        # augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        shear=10.0,
        flipud=0.05,
        fliplr=0.5,
        mosaic=0.3,
        close_mosaic=15,
    )

    # Extract final metrics
    metrics = results.results_dict
    map50 = metrics.get("metrics/mAP50(B)", None)
    map5095 = metrics.get("metrics/mAP50-95(B)", None)
    precision = metrics.get("metrics/precision(B)", None)
    recall = metrics.get("metrics/recall(B)", None)

    # Save CSV
    with open(RESULTS_CSV, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            stage_name,
            lr,
            freeze,
            map50,
            map5095,
            precision,
            recall
        ])

print("\nAll progressive unfreezing stages completed.")