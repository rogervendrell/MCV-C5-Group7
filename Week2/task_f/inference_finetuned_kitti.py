import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

from sam_utils import load_sam
from segment_anything import SamPredictor

# Paths
IMG_DIR = Path("/ghome/group07/MCV-C5-Group7/Week2/task_f/dataset/images/")
IMAGE_FILES = [
    "00001248.jpg",
    "00002204.jpg",
    "00002816.jpg",
    "00004446.jpg",
    "00004671.jpg",
    "00009650.jpg"
]

GT_FILE = IMG_DIR.parent / "annotations.json"
OUT_FILE = "/ghome/group07/MCV-C5-Group7/Week2/task_f/sam_predictions.json"
CHECKPOINT = "/ghome/group07/MCV-C5-Group7/Week2/task_e/sam_kitti_decoder_3.pth"
OUT_IMG_DIR = IMG_DIR.parent.parent / "inference_visualizations"
OUT_IMG_DIR.mkdir(exist_ok=True, parents=True)

# Utility function
def get_points_from_coco_annotations(anns):
    points = []
    labels = []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        cx = int(x + w / 2)
        cy = int(y + h * 0.4)  # slightly above center
        points.append([cx, cy])
        labels.append(1)
    return np.array(points), np.array(labels)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    coco = COCO(str(GT_FILE))
    predictions = []

    # --- Load SAM and finetuned decoder ---
    sam_model = load_sam("vit_b", "/ghome/group07/MCV-C5-Group7/Week2/task_e/sam_vit_b_01ec64.pth", device)
    sam_model.mask_decoder.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    sam_model.eval()
    predictor = SamPredictor(sam_model)

    for img_file in tqdm(IMAGE_FILES, desc="Running SAM inference"):
        # Find numeric COCO image id
        img_ids = coco.getImgIds()
        img_id = None
        for i in img_ids:
            if coco.imgs[i]["file_name"] == img_file:
                img_id = i
                break
        if img_id is None:
            print(f"Skipping {img_file}: not in COCO annotations")
            continue

        # Load image
        img_info = coco.loadImgs(img_id)[0]
        img_path = IMG_DIR / img_info["file_name"]
        if not img_path.exists():
            print(f"Image not found: {img_path}")
            continue
        image = np.array(Image.open(img_path).convert("RGB"))

        # Load annotations
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        if len(anns) == 0:
            continue

        # Generate points for predictor
        points, labels = get_points_from_coco_annotations(anns)
        if len(points) == 0:
            continue

        # Set the image for predictor
        predictor.set_image(image)

        # Predict masks
        masks, scores, logits = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=False
        )

        # Create a copy for visualization
        vis_image = image.copy()

        # Overlay predicted masks in red
        for mask in masks:
            red_mask = np.zeros_like(vis_image)
            red_mask[:, :, 0] = 255  # red channel
            vis_image = np.where(mask[:, :, None], red_mask, vis_image)

        # Draw GT bounding boxes in green
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(vis_image)
        for ann in anns:
            x, y, w, h = ann["bbox"]
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='green', facecolor='none')
            ax.add_patch(rect)

        ax.axis('off')
        out_path = OUT_IMG_DIR / img_file
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Convert each mask to COCO RLE
        for ann, mask in zip(anns, masks):
            binary_mask = mask.astype(np.uint8)
            if binary_mask.sum() == 0:
                continue

            rle = mask_utils.encode(np.asfortranarray(binary_mask))
            rle["counts"] = rle["counts"].decode("utf-8")
            area = int(mask_utils.area(rle))
            bbox = mask_utils.toBbox(rle).tolist()

            predictions.append({
                "image_id": img_id,
                "category_id": ann["category_id"],
                "segmentation": rle,
                "score": float(scores[0]),
                "area": area,
                "bbox": bbox
            })

    # Save predictions
    with open(OUT_FILE, "w") as f:
        json.dump(predictions, f)

    print("Finished SAM inference")
    print("Predictions saved to:", OUT_FILE)
    print(f"Visualizations saved in: {OUT_IMG_DIR}")
    print("Total predictions:", len(predictions))


if __name__ == "__main__":
    main()