import os
import json
import random
import cv2
import numpy as np
from pycocotools import mask as maskUtils

IMAGE_DIR = "dataset/images"
ANNOTATIONS_FILE = "dataset/annotations.json"
SAM_FILE = "sam_predictions.json"
OUTPUT_DIR = "visualise"


def load_coco_annotations(path):
    with open(path) as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}

    anns_by_image = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    return images, anns_by_image


def load_sam_predictions(path):
    with open(path) as f:
        preds = json.load(f)

    preds_by_image = {}
    for p in preds:
        preds_by_image.setdefault(p["image_id"], []).append(p)

    return preds_by_image


def draw_boxes(img, annotations):
    for ann in annotations:
        x, y, w, h = map(int, ann["bbox"])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


def draw_masks(img, predictions):
    overlay = img.copy()

    for pred in predictions:
        rle = pred["segmentation"]
        mask = maskUtils.decode(rle)

        color = np.array([0, 0, 255])  # red
        overlay[mask == 1] = overlay[mask == 1] * 0.5 + color * 0.5

    return overlay.astype(np.uint8)


def main(N=5):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images, gt_annotations = load_coco_annotations(ANNOTATIONS_FILE)
    sam_predictions = load_sam_predictions(SAM_FILE)

    image_ids = list(images.keys())
    chosen_ids = random.sample(image_ids, min(N, len(image_ids)))

    for image_id in chosen_ids:
        img_info = images[image_id]
        img_path = os.path.join(IMAGE_DIR, img_info["file_name"])

        img = cv2.imread(img_path)
        if img is None:
            continue

        # Draw GT bounding boxes
        draw_boxes(img, gt_annotations.get(image_id, []))

        # Draw SAM masks
        img = draw_masks(img, sam_predictions.get(image_id, []))

        # Save result
        out_path = os.path.join(OUTPUT_DIR, img_info["file_name"])
        cv2.imwrite(out_path, img)

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main(N=5)