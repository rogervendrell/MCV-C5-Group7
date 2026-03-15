import os
import json
import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as mask_utils

ds = load_dataset("biglam/european_art")

out_img_dir = "/ghome/group07/MCV-C5-Group7/Week2/task_f/dataset/images/"
os.makedirs(out_img_dir, exist_ok=True)
ann_path = "/ghome/group07/MCV-C5-Group7/Week2/task_f/dataset/annotations.json"

annotations = []
images = []

ann_id = 1
img_id = 1

for split in ds.keys():
    for sample in tqdm(ds[split], desc=f"Processing {split}", total=len(ds[split])):
        ann = json.loads(sample["annotations"])

        category_map = {c["id"]: c["name"] for c in ann["categories"]}

        person_ids = [cid for cid, name in category_map.items() if name == "person"]
        if not person_ids:
            continue

        image = sample["image"]
        width, height = image.size

        file_name = f"{img_id:08d}.jpg"
        img_path = os.path.join(out_img_dir, file_name)

        image.save(img_path)

        images.append({
            "id": img_id,
            "file_name": file_name,
            "height": height,
            "width": width
        })

        for a in ann["annotations"]:
            if a["category_id"] not in person_ids:
                continue
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": a["bbox"],
                "area": a["area"],
                "iscrowd": 0
            })
            ann_id += 1

        img_id += 1

coco = {
    "images": images,
    "annotations": annotations,
    "categories": [
        {"id": 1, "name": "person"}
    ]
}

with open(ann_path, "w") as f:
    json.dump(coco, f)

print("Dataset conversion finished")
print("Images:", len(images))
print("Annotations:", len(annotations))