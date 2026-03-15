import os
import json
import random
import cv2
import numpy as np
from pycocotools import mask as maskUtils

images_root = "/export/home/mcv/datasets/C5/KITTI-MOTS/training/image_02"
predictions_file = "predictions/sam_yolo_predictions_0.01conf.json"
output_dir = "visualizations"

os.makedirs(output_dir, exist_ok=True)

# Load predictions
with open(predictions_file) as f:
    preds = json.load(f)

# Group predictions by image_id
preds_by_image = {}
for p in preds:
    preds_by_image.setdefault(p["image_id"], []).append(p)

# Random sample of 5 images
sample_ids = random.sample(list(preds_by_image.keys()), 5)

for image_id in sample_ids:

    seq = image_id // 100000
    frame = image_id % 100000

    img_path = os.path.join(
        images_root,
        f"{seq:04d}",
        f"{frame:06d}.png"
    )

    if not os.path.exists(img_path):
        print("Missing:", img_path)
        continue

    image = cv2.imread(img_path)
    overlay = image.copy()

    for pred in preds_by_image[image_id]:

        mask = maskUtils.decode(pred["segmentation"])
        color = np.random.randint(0, 255, 3)

        overlay[mask == 1] = color

    vis = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

    out_path = os.path.join(
        output_dir,
        f"{seq:04d}_{frame:06d}.png"
    )

    cv2.imwrite(out_path, vis)
    print("Saved:", out_path)