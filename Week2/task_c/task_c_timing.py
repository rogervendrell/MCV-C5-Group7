import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import json
import time

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from transformers import AutoModel, AutoProcessor
from pycocotools import mask as mask_utils
from tqdm import tqdm


# PATHS
IMAGES_DIR = Path("/ghome/mcv/datasets/C5/KITTI-MOTS/training/image_02/")
YOLO_BASE = Path("/ghome/group07/MCV-C5-Group7/Week2/task_c/yolo_bboxes")
OUT_DIR = Path("/ghome/group07/MCV-C5-Group7/Week2/SAMOutput/c_yolo_bbox")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREDS_DIR = Path("/ghome/group07/MCV-C5-Group7/Week2/task_c/predictions")
PREDS_DIR.mkdir(parents=True, exist_ok=True)

VISUALIZE = False

CONF_LEVELS = ["0.01", "0.20", "0.50", "0.70", "0.90"]

# timing settings
MAX_TIMED_IMAGES = 100
times_ms = []


# LOAD PRECOMPUTED BOXES
def load_yolo_boxes(img_path, yolo_boxes_dir):
    seq = img_path.parent.name
    bbox_file = yolo_boxes_dir / seq / f"{img_path.stem}.txt"

    class_map = {"car": 1, "person": 2}

    if not bbox_file.exists():
        return []

    classes, scores, boxes = [], [], []

    with open(bbox_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            classes.append(class_map[parts[0]])
            scores.append(float(parts[1]))
            boxes.append(list(map(float, parts[2:])))

    return classes, scores, boxes


# VISUALIZATION
def visualize_masks(image, masks, boxes, output_path):

    plt.figure(figsize=(15,6))
    plt.imshow(image)
    ax = plt.gca()

    num_masks = masks.shape[0]
    colors = np.random.rand(num_masks,3)

    for i in range(num_masks):

        m = masks[i][0].cpu().numpy()

        mask_image = np.zeros((m.shape[0], m.shape[1], 4))
        mask_image[m > 0] = np.concatenate([colors[i], [0.5]])
        ax.imshow(mask_image)

        box = boxes[i]

        rect = mpatches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            fill=False,
            edgecolor="lime",
            linewidth=2
        )

        ax.add_patch(rect)

    plt.axis("off")
    plt.title("SAM with YOLO bounding box prompts")
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()


# MAIN
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SAM
    model = AutoModel.from_pretrained("facebook/sam-vit-base").to(device)
    processor = AutoProcessor.from_pretrained("facebook/sam-vit-base")

    all_img_seqs = sorted(os.listdir(IMAGES_DIR))
    img_paths = []

    for seq in ["0000"]:
        seq_img_dir = IMAGES_DIR / seq
        images = sorted(seq_img_dir.glob("*.png"))
        img_paths.extend(images)

    for conf in CONF_LEVELS:

        print(f"\nRunning SAM with YOLO confidence {conf}")

        YOLO_BBOXES = YOLO_BASE / f"conf{conf}"

        coco_results = []
        ann_id = 1

        timed_count = 0

        for img_p in tqdm(img_paths, desc=f"Processing images (conf {conf})", total=len(img_paths)):

            raw_image = Image.open(img_p).convert("RGB")

            # LOAD YOLO BOXES
            classes, scores, boxes = load_yolo_boxes(img_p, YOLO_BBOXES)

            if len(boxes) == 0:
                continue

            inputs = processor(
                images=raw_image,
                input_boxes=[boxes],
                return_tensors="pt"
            ).to(device)

            # ---- START TIMING ----
            if timed_count < MAX_TIMED_IMAGES:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start_time = time.time()

            with torch.no_grad():
                outputs = model(**inputs)

            processed_masks = processor.post_process_masks(
                outputs.pred_masks,
                inputs["original_sizes"],
                inputs["reshaped_input_sizes"]
            )[0]

            if timed_count < MAX_TIMED_IMAGES:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed_ms = (time.time() - start_time) * 1000
                times_ms.append(elapsed_ms)
                timed_count += 1
            # ---- END TIMING ----

            seq = int(img_p.parent.name)
            frame = int(img_p.stem)
            img_id = seq * 100000 + frame

            assert len(processed_masks) == len(boxes) == len(classes) == len(scores)

            for mask_tensor, category_id, score in zip(processed_masks, classes, scores):

                mask = mask_tensor[0].cpu().numpy()
                mask = (mask > 0).astype(np.uint8)

                rle = mask_utils.encode(np.asfortranarray(mask))
                rle["counts"] = rle["counts"].decode("utf-8")

                coco_results.append({
                    "image_id": img_id,
                    "category_id": category_id,
                    "segmentation": rle,
                    "score": score
                })

                ann_id += 1

            # PRINT TIMING RESULTS
            mean_time = np.mean(times_ms)
            std_time = np.std(times_ms)
            print("\n----- Inference Timing -----")
            print(f"Images timed: {len(times_ms)}")
            print(f"Mean inference time: {mean_time:.2f} ms/image")
            print(f"Std deviation: {std_time:.2f} ms")

            if VISUALIZE:
                filename = os.path.basename(img_p)
                save_path = OUT_DIR / f"vis_conf{conf}_{filename}"

                visualize_masks(
                    raw_image,
                    processed_masks,
                    boxes,
                    save_path
                )

        # SAVE PREDICTIONS
        pred_file = PREDS_DIR / f"sam_yolo_predictions_{conf}conf.json"

        with open(pred_file, "w") as f:
            json.dump(coco_results, f)

        print(f"Saved {len(coco_results)} predictions to {pred_file}")