import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoModel, AutoProcessor

from pycocotools.coco import COCO
from pycocotools import mask as mask_utils


# Paths
IMG_DIR = Path("/ghome/group07/MCV-C5-Group7/Week2/task_f/dataset/images/")
ANN_FILE = IMG_DIR.parent / "annotations.json"

OUT_PRED_FILE = "/ghome/group07/MCV-C5-Group7/Week2/task_f/sam_predictions.json"


# centroid point from COCO bboxes
def get_points_from_coco_annotations(anns):
    points = []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        cx = int(x + w / 2)
        cy = int(y + h * 0.4) # no ben be al mig sino una mica mes amunt 
        points.append([cx, cy])

    return points


# Main
def main():
    # Load annotations
    coco = COCO(str(ANN_FILE))
    img_ids = coco.getImgIds()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load SAM
    model = AutoModel.from_pretrained("facebook/sam-vit-base").to(device)
    processor = AutoProcessor.from_pretrained("facebook/sam-vit-base")

    predictions = []

    for img_id in tqdm(img_ids, desc="Running SAM inference"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = IMG_DIR / img_info["file_name"]

        if not img_path.exists():
            continue

        raw_image = Image.open(img_path).convert("RGB")

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        if len(anns) == 0:
            continue

        # Generate prompt points
        points = get_points_from_coco_annotations(anns)
        MAX_MASKS = 100
        if len(points) > MAX_MASKS:
            points = points[:MAX_MASKS]

        if len(points) == 0:
            continue

        # SAM expects nested structure
        input_points = [[[p] for p in points]]
        input_labels = [[[1] for _ in points]]

        inputs = processor(
            images=raw_image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Convert masks to image resolution
        processed_masks = processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"]
        )[0]

        # SAM returns 3 masks per prompt (take best one)
        best_masks = processed_masks[:, 0]

        for mask in best_masks:
            m = mask.cpu().numpy().astype(np.uint8)

            if m.sum() == 0:
                continue

            rle = mask_utils.encode(np.asfortranarray(m))
            rle["counts"] = rle["counts"].decode("utf-8")

            area = int(mask_utils.area(rle))
            bbox = mask_utils.toBbox(rle).tolist()

            predictions.append({
                "image_id": img_id,
                "category_id": 1,
                "segmentation": rle,
                "score": 1.0,
                "area": area,
                "bbox": bbox
            })

    # Save predictions
    with open(OUT_PRED_FILE, "w") as f:
        json.dump(predictions, f)

    print("Finished SAM inference")
    print("Predictions saved to:", OUT_PRED_FILE)
    print("Total predictions:", len(predictions))


if __name__ == "__main__":
    main()