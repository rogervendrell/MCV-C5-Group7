import json
from pathlib import Path
from pycocotools import mask as mask_utils

INSTANCES_DIR = Path("/ghome/mcv/datasets/C5/KITTI-MOTS/instances_txt")
GT_PATH = Path("/ghome/group07/MCV-C5-Group7/Week2/task_c/ground_truth/kitti_mots_gt_coco.json")

images = []
annotations = []

ann_id = 1

for seq_file in sorted(INSTANCES_DIR.glob("*.txt")):
    seq = int(seq_file.stem)

    with open(seq_file) as f:
        for line in f:
            parts = line.strip().split()

            frame = int(parts[0])
            track_id = int(parts[1])
            category_id = int(parts[2])
            h = int(parts[3])
            w = int(parts[4])
            counts = parts[5]

            image_id = seq * 100000 + frame

            images.append({
                "id": image_id,
                "file_name": f"{seq:04d}/{frame:06d}.png",
                "height": h,
                "width": w
            })

            rle = {
                "size": [h, w],
                "counts": counts
            }

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": rle,
                "area": float(mask_utils.area(rle)),
                "bbox": mask_utils.toBbox(rle).tolist(),
                "iscrowd": 0
            })

            ann_id += 1


coco_gt = {
    "images": images,
    "annotations": annotations,
    "categories": [
        {"id": 1, "name": "car"},
        {"id": 2, "name": "pedestrian"}
    ]
}

with open(GT_PATH, "w") as f:
    json.dump(coco_gt, f)

print("Saved COCO GT")