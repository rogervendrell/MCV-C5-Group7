import json
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils

# -------- PATHS --------
GT_FILE = "/ghome/group07/MCV-C5-Group7/Week2/task_c/ground_truth/kitti_mots_gt_coco_filtered.json"
PRED_FILE = "sam_predictions_epoch5.json"
DATASET_DIR = "/ghome/group07/mcv/datasets/C5/KITTI-MOTS/training/image_02"

OUTPUT_DIR = "visualizations2"
NUM_IMAGES = 10


def random_color():
    return np.random.rand(3)


def overlay_mask(image, mask, color, alpha=0.5):
    colored_mask = np.zeros_like(image, dtype=np.float32)

    for c in range(3):
        colored_mask[:, :, c] = mask * color[c] * 255

    return np.where(mask[:, :, None], image * (1 - alpha) + colored_mask * alpha, image)


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    coco = COCO(GT_FILE)

    with open(PRED_FILE) as f:
        preds = json.load(f)

    # group predictions per image
    preds_by_image = {}
    for p in preds:
        preds_by_image.setdefault(p["image_id"], []).append(p)

    img_ids = list(preds_by_image.keys())
    sampled_ids = random.sample(img_ids, min(NUM_IMAGES, len(img_ids)))

    print(f"Saving {len(sampled_ids)} visualizations...")

    for img_id in sampled_ids:

        img_info = coco.loadImgs(img_id)[0]
        img_path = f"{DATASET_DIR}/{img_info['file_name']}"

        image = np.array(Image.open(img_path).convert("RGB"))
        vis = image.copy()

        for pred in preds_by_image[img_id]:

            mask = mask_utils.decode(pred["segmentation"])
            color = random_color()

            vis = overlay_mask(vis, mask, color)

        out_path = os.path.join(
            OUTPUT_DIR,
            img_info["file_name"].replace("/", "_")
        )

        plt.figure(figsize=(10, 6))
        plt.imshow(vis.astype(np.uint8))
        plt.axis("off")

        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    print("Saved results to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()