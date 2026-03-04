import os
import csv
from datetime import datetime
from ultralytics import YOLO

# Slurm variables
job_name = os.environ.get("SLURM_JOB_NAME", "nojname")
user = os.environ.get("USER", "nouser")
job_id = os.environ.get("SLURM_JOB_ID", "nojobid")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Config
EPOCHS = 25
LEARNING_RATES = [0.0001]
AUGMENTATION_OPTIONS = [True]
BATCH_SIZES = [8, 16, 32]
WEIGHT_DECAY = 0.0005
OPTIMIZER = "AdamW"

PROJECT_NAME = "task_e_bs"
RESULTS_CSV = f"/ghome/group07/MCV-C5-Group7/ultralytics/output/task_e/{job_name}_{user}_{job_id}_{timestamp}.csv"

augmentations = {
    "hsv_h":     [0.015, 0.0],
    "hsv_s":     [0.7,   0.0],
    "hsv_v":     [0.4,   0.0],
    "translate": [0.1,   0.0],
    "shear":     [10.0,  0.0],
    "flipud":    [0.05,  0.0],
    "fliplr":    [0.5,   0.0],
    "mosaic":    [0.3,   0.0],
}

# CSV header
if not os.path.exists(RESULTS_CSV):
    with open(RESULTS_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_name",
            "lr",
            "augmentation",
            "mAP50",
            "mAP50-95",
            "precision",
            "recall"
        ])

# Experiment
for lr in LEARNING_RATES:
    for DATA_AUGMENTATION in AUGMENTATION_OPTIONS:
        for BATCH_SIZE in BATCH_SIZES:
            choose = 0 if DATA_AUGMENTATION else 1

            run_name = f"lr{lr}_aug{DATA_AUGMENTATION}"

            print(f"\n===== Starting experiment: {run_name} =====\n")

            model = YOLO("/ghome/group07/MCV-C5-Group7/ultralytics/yolo26n.pt")

            results = model.train(
                data="/ghome/group07/MCV-C5-Group7/ultralytics/kitti.yaml",
                epochs=EPOCHS,
                imgsz=640,
                batch=BATCH_SIZE,
                lr0=lr,
                optimizer=OPTIMIZER,
                weight_decay=WEIGHT_DECAY,
                project=PROJECT_NAME,   # main folder
                name=run_name,          # subfolder name
                exist_ok=True,
                save=True,
                cache=True,
                workers=4,
                seed=42,
                val=True,
                plots=True,
                # augmentation
                hsv_h=augmentations["hsv_h"][choose],
                hsv_s=augmentations["hsv_s"][choose],
                hsv_v=augmentations["hsv_v"][choose],
                translate=augmentations["translate"][choose],
                shear=augmentations["shear"][choose],
                flipud=augmentations["flipud"][choose],
                fliplr=augmentations["fliplr"][choose],
                mosaic=augmentations["mosaic"][choose],
                close_mosaic=15,
            )

            # Extract final metrics
            metrics = results.results_dict

            map50 = metrics.get("metrics/mAP50(B)", None)
            map5095 = metrics.get("metrics/mAP50-95(B)", None)
            precision = metrics.get("metrics/precision(B)", None)
            recall = metrics.get("metrics/recall(B)", None)

            # Save to CSV
            with open(RESULTS_CSV, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    run_name,
                    lr,
                    DATA_AUGMENTATION,
                    map50,
                    map5095,
                    precision,
                    recall
                ])

print("\nAll experiments completed.")