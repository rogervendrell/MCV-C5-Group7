import os
import csv
from datetime import datetime
from ultralytics import YOLO
from albumentations_aug import get_transforms

# Slurm variables
job_name = os.environ.get("SLURM_JOB_NAME", "nojname")
user = os.environ.get("USER", "nouser")
job_id = os.environ.get("SLURM_JOB_ID", "nojobid")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Config
EPOCHS = 25
LEARNING_RATES = [0.01, 0.001, 0.0001,] # 0.00001,  
AUGMENTATION_OPTIONS = [True, False]
WEIGHT_DECAY = 0.0005
OPTIMIZER = "AdamW"

PROJECT_NAME = "task_e_experiment_albumentations"
RESULTS_CSV = f"/ghome/group07/MCV-C5-Group7/ultralytics/output/task_e/{job_name}_{user}_{job_id}_{timestamp}.csv"

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

        run_name = f"lr{lr}_aug{DATA_AUGMENTATION}"

        print(f"\n===== Starting experiment: {run_name} =====\n")

        model = YOLO("/ghome/group07/MCV-C5-Group7/ultralytics/yolo26n.pt")

        results = model.train(
            data="/ghome/group07/MCV-C5-Group7/ultralytics/kitti.yaml",
            epochs=EPOCHS,
            imgsz=640,
            batch=0.90,
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
            augmentations=get_transforms(DATA_AUGMENTATION),
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