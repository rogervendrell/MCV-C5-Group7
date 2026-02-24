import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools import mask as maskUtils


# Paths
ROOT = "/ghome/group07/mcv/datasets/C5/KITTI-MOTS"
OUT_ROOT = "/ghome/group07/MCV-C5-Group7/ultralytics/dataset"

TRAIN_IMG_ROOT = os.path.join(ROOT, "training/image_02")
INST_TXT_ROOT = os.path.join(ROOT, "instances_txt")

VAL_SEQS = {'0002','0006','0007','0008','0010','0013','0014','0016','0018'}

for split in ['train', 'val']:
    os.makedirs(os.path.join(OUT_ROOT, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, 'labels', split), exist_ok=True)

# YAML
YAML_CONTENT = f"""
path: /ghome/group07/MCV-C5-Group7/ultralytics/dataset
train: images/train
val: images/val

names:
  0: car
  1: person
"""

# RLE
def rle_to_bbox(rle_counts, height, width):
    rle = {
        'size': [height, width],
        'counts': rle_counts.encode('utf-8')
    }
    bbox = maskUtils.toBbox(rle)
    # x, y, w, h
    return bbox


# Main
if __name__ == "__main__":
    for seq_file in tqdm(os.listdir(INST_TXT_ROOT)):
    
        seq_id = seq_file.replace(".txt", "")
        split = "val" if seq_id in VAL_SEQS else "train"

        seq_img_folder = os.path.join(TRAIN_IMG_ROOT, seq_id)
        seq_txt_path = os.path.join(INST_TXT_ROOT, seq_file)

        with open(seq_txt_path, 'r') as f:
            lines = f.readlines()

        # Group objects per frame
        frame_dict = {}

        for line in lines:
            parts = line.strip().split(" ")

            frame_id = int(parts[0])
            obj_id = int(parts[1])
            class_id_kitti = int(parts[2])
            height = int(parts[3])
            width = int(parts[4])
            rle_counts = parts[5]

            # Ignore region TODO: mirar què fer-ne
            if obj_id == 10000:
                continue

            # Only valid classes
            if class_id_kitti not in [1, 2]:
                print("Invalid class with classid", class_id)
                continue

            # Kitti -> yolo
            # car: 0, pedestrian: 1
            class_id = class_id_kitti - 1

            bbox = rle_to_bbox(rle_counts, height, width)

            if frame_id not in frame_dict:
                frame_dict[frame_id] = []

            frame_dict[frame_id].append((class_id, bbox, height, width))

        # Process frames
        for frame_id, objects in frame_dict.items():

            img_name = f"{frame_id:06d}.png"
            img_path = os.path.join(seq_img_folder, img_name)

            if not os.path.exists(img_path):
                continue

            image = cv2.imread(img_path)
            h_img, w_img = image.shape[:2]

            new_img_name = f"{seq_id}_{img_name}"

            out_img_path = os.path.join(OUT_ROOT, 'images', split, new_img_name)
            shutil.copy(img_path, out_img_path)

            out_label_path = os.path.join(OUT_ROOT, 'labels', split, new_img_name.replace(".png", ".txt"))

            with open(out_label_path, 'w') as lf:

                for class_id, bbox, height, width in objects:

                    x, y, w, h = bbox

                    cx = x + w / 2
                    cy = y + h / 2
                    cx /= w_img
                    cy /= h_img
                    w /= w_img
                    h /= h_img

                    lf.write(f"{class_id} {cx} {cy} {w} {h}\n")
    
    # yaml
    with open(os.path.join(OUT_ROOT, "kitti.yaml"), "w") as f:
        f.write(YAML_CONTENT.strip())
