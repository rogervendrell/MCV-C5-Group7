import os
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import torch
import matplotlib.pyplot as plt
import random

from sklearn.cluster import KMeans

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# Paths
IMAGES_DIR = Path("/ghome/mcv/datasets/C5/KITTI-MOTS/training/image_02/")
OUT_DIR = Path("/ghome/group07/MCV-C5-Group7/Week2/task_h/visualise/")

SAM_CHECKPOINT = "/ghome/group07/MCV-C5-Group7/Week2/task_h/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# Mask embedding extraction
def get_mask_embedding_using_patch_embeddings(mask, enc_emb):

    # Convert 1024x1024 mask -> 64x64 patch grid
    split_mask = np.array(np.split(mask, 64, axis=-1))
    split_mask = np.array(np.split(split_mask, 64, axis=-2))

    split_mask = split_mask * 1

    split_mask = np.sum(split_mask, axis=-1)
    split_mask = np.sum(split_mask, axis=-1)

    patch_locations = np.where(split_mask > 1)

    patch_embeddings = enc_emb[patch_locations]

    if len(patch_embeddings) == 0:
        return None

    mask_embedding = patch_embeddings.mean(axis=0)

    return mask_embedding


# Visualization
def visualize_semantic(image, masks, labels, output_path):
    plt.figure(figsize=(12,6))
    plt.imshow(image)

    ax = plt.gca()
    unique_labels = np.unique(labels)
    colors = np.random.rand(len(unique_labels),3)
    
    for i, cluster_id in enumerate(unique_labels):
        if cluster_id == -1:
            continue

        seg_mask = np.zeros(masks[0]['segmentation'].shape)
        for idx in np.where(labels == cluster_id)[0]:
            seg_mask += masks[idx]['segmentation']

        seg_mask = seg_mask > 0
        color = np.concatenate([colors[i], [0.5]])
        overlay = np.zeros((seg_mask.shape[0], seg_mask.shape[1],4))
        overlay[seg_mask] = color
        ax.imshow(overlay)

    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()


# Main
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=51,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100
    )

    all_sequences = sorted(os.listdir(IMAGES_DIR))

    for seq in all_sequences:

        seq_dir = IMAGES_DIR / seq
        images = sorted(seq_dir.glob("*.png"))
        random.shuffle(images)

        for img_path in images:

            print("Processing:", img_path)

            image = cv2.imread(str(img_path))
            original_h, original_w = image.shape[:2]
            image = cv2.resize(image, (1024,1024))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Extract encoder embeddings
            mask_generator.predictor.set_image(image_rgb)

            enc_emb = mask_generator.predictor.features
            enc_emb = enc_emb.to("cpu").numpy()
            enc_emb = enc_emb[0].transpose((1,2,0))

            # Automatic mask generation
            masks = mask_generator.generate(image_rgb)
            masks = sorted(masks, key=lambda x: x['area'], reverse=True)

            print("Number of masks:", len(masks))

            # Compute mask embeddings
            mask_embeddings = []
            valid_indices = []

            for i, m in enumerate(masks):
                mask = m["segmentation"]
                emb = get_mask_embedding_using_patch_embeddings(mask, enc_emb)
                if emb is not None:
                    mask_embeddings.append(emb)
                    valid_indices.append(i)

            mask_embeddings = np.array(mask_embeddings)
            print("Embedding shape:", mask_embeddings.shape)

            # --- KMeans Clustering ---
            kmeans = KMeans(n_clusters=4, random_state=42).fit(mask_embeddings)
            labels = kmeans.labels_
            print("Clusters:", np.unique(labels))

            # Visualization
            save_path = OUT_DIR / f"semantic_{img_path.name}"
            visualize_semantic(
                image_rgb,
                masks,
                labels,
                save_path
            )