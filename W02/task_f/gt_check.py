import os
import json
import random
import cv2
import numpy as np
from pycocotools import mask as mask_utils

DATASET_DIR = "/ghome/group07/MCV-C5-Group7/Week2/task_f/dataset"
IMG_DIR = os.path.join(DATASET_DIR, "images")
ANN_PATH = os.path.join(DATASET_DIR, "annotations.json")

NUM_SAMPLES = 5
OUTPUT_DIR = "gt_check_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load annotations
with open(ANN_PATH, "r") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]

# Build image_id -> annotations map
ann_map = {}
for ann in annotations:
    ann_map.setdefault(ann["image_id"], []).append(ann)

# Pick random images
sampled_images = random.sample(images, NUM_SAMPLES)

for img_info in sampled_images:

    img_id = img_info["id"]
    file_name = img_info["file_name"]

    img_path = os.path.join(IMG_DIR, file_name)
    img = cv2.imread(img_path)

    if img is None:
        print("Failed to load", img_path)
        continue

    anns = ann_map.get(img_id, [])
    breakpoint()
    for ann in anns:
        # Decode mask
        rle = ann["segmentation"]
        rle["counts"] = rle["counts"].encode("utf-8")
        mask = mask_utils.decode(rle)

        # Draw mask
        color = np.array([0, 255, 0], dtype=np.uint8)
        img[mask == 1] = img[mask == 1] * 0.5 + color * 0.5

        # Draw bbox
        x, y, w, h = map(int, ann["bbox"])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    out_path = os.path.join(OUTPUT_DIR, file_name)
    cv2.imwrite(out_path, img)

    print("Saved:", out_path)

print("Done.")