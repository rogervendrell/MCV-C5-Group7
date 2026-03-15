import os
import numpy as np
import random
from pathlib import Path
from PIL import Image
from scipy.ndimage import label

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import time
import torch

from transformers import AutoModel, AutoProcessor, pipeline

# Paths to the validation split (update if you change your dataset layout)
IMAGES_DIR = Path("/ghome/mcv/datasets/C5/KITTI-MOTS/training/image_02/")
MASKS_DIR = Path("/ghome/mcv/datasets/C5/KITTI-MOTS/instances/")
OUT_DIR_POINTS = Path("/ghome/group07/MCV-C5-Group7/Week2/SAMOutput/a_points")
OUT_DIR_BBOX = Path("/ghome/group07/MCV-C5-Group7/Week2/SAMOutput/a_bbox")
OUT_DIR_GRID = Path("/ghome/group07/MCV-C5-Group7/Week2/SAMOutput/a_grid")

np.random.seed(42)

"""-------------------------------------------------------------------------"""
def visualize_and_save(image, masks, points, point_types, boxes, boxes_types, output_path):
    plt.figure(figsize=(15, 6)) # Wide for KITTI aspect ratio
    plt.imshow(image)
    
    ax = plt.gca()
    num_objects = masks.shape[0]
    
    # Generate random colors for the masks
    colors = np.random.rand(num_objects, 3)
    
    for i in range(num_objects):
        m = masks[i][0].cpu().numpy() if hasattr(masks[i], 'cpu') else masks[i][0]
        
        # Create a colored overlay for the mask
        mask_image = np.zeros((m.shape[0], m.shape[1], 4)) # RGBA
        mask_image[m > 0] = np.concatenate([colors[i], [0.5]]) # 0.5 is transparency
        ax.imshow(mask_image)
        
        
        # Draw the point
        p = points[i]
        p_type = point_types[i]
        color = 'lime' if p_type == 'standard' else 'red'
        marker = 'o' if p_type == 'standard' else 'P' # 'P' is a thick plus sign
        
        ax.scatter(p[0], p[1], color=color, edgecolors='black', 
                   marker=marker, s=60, lw=1, zorder=5)

        # Draw the Bounding Box
        box = boxes[i]
        b_type = boxes_types[i]
        color = 'lime' if b_type == 'standard' else 'red'
        rect = mpatches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            fill=False, edgecolor=color, linewidth=1, linestyle='--', alpha=0.7
        )
        ax.add_patch(rect)
        
    # Add a custom legend
    std_patch = mpatches.Patch(color='lime', label='Standard Object Point')
    ref_patch = mpatches.Patch(color='red', label='Crowd (10000) Point')
    plt.legend(handles=[std_patch, ref_patch], loc='upper right')
    
    plt.axis('off')
    plt.title(f"SAM 32 Grid Inference", fontsize=15)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
"""-------------------------------------------------------------------------"""
def visualize_everything(image, pipeline_output, output_path):
    plt.figure(figsize=(15, 6))
    plt.imshow(image)
    ax = plt.gca()

    # 1. Convert everything to numpy immediately to avoid library conflicts
    masks = []
    for m in pipeline_output["masks"]:
        # If it's a tensor, move to CPU and convert to numpy
        if torch.is_tensor(m):
            masks.append(m.detach().cpu().numpy())
        else:
            masks.append(np.array(m))
            
    scores = pipeline_output["scores"]
    
    # 2. Sort masks by area (sum of pixels)
    # We draw larger masks (road/sky) first, and smaller ones (cars/people) last
    # This prevents big background masks from hiding the small objects
    sorted_masks = sorted(zip(masks, scores), 
                          key=lambda x: x[0].sum(), # Standard numpy sum
                          reverse=True)

    # 3. Plotting
    for i, (m, score) in enumerate(sorted_masks):
        # Generate a random color with 0.5 alpha
        color = np.concatenate([np.random.random(3), [0.5]])
        
        # Create the RGBA overlay
        mask_image = np.zeros((m.shape[0], m.shape[1], 4))
        mask_image[m > 0] = color # Use boolean indexing
        ax.imshow(mask_image)

    plt.axis('off')
    plt.title(f"SAM 'Everything' Mode: 32 points per batch", fontsize=15)
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close()


"""-------------------------------------------------------------------------"""
def get_points_with_metadata(mask_path):
    mask = np.array(Image.open(mask_path))
    points = []
    types = []
    
    # 1. Standard IDs
    standard_ids = np.unique(mask)
    # Filter for valid object IDs (e.g., cars/pedestrians)
    standard_ids = standard_ids[(standard_ids > 0) & (standard_ids < 10000)]
    
    for obj_id in standard_ids:
        y_coords, x_coords = np.where(mask == obj_id)
        
        if len(x_coords) > 0:
            # Pick a random index from the available pixels
            idx = np.random.randint(0, len(x_coords))
            points.append([x_coords[idx], y_coords[idx]])
            types.append('standard')

    # 2. Refined 10000 IDs (Crowd/Ignore regions)
    if 10000 in mask:
        crowd_mask = (mask == 10000).astype(int)
        labeled_crowd, num_features = label(crowd_mask)
        
        for i in range(1, num_features + 1):
            y_c, x_c = np.where(labeled_crowd == i)
            center_x = len(x_c) // 2
            center_y = len(y_c) // 2
            points.append([x_c[center_x], y_c[center_y]])
            types.append('refined')
                
    return points, types

"""-------------------------------------------------------------------------"""
def get_boxes_with_metadata(mask_path):
    mask = np.array(Image.open(mask_path))
    boxes = []
    types = []
    
    # 1. Standard IDs
    standard_ids = np.unique(mask)
    standard_ids = standard_ids[(standard_ids > 0) & (standard_ids < 10000)]
    
    for obj_id in standard_ids:
        y_coords, x_coords = np.where(mask == obj_id)
        if len(x_coords) > 0:
            # Box format: [xmin, ymin, xmax, ymax]
            box = [np.min(x_coords), np.min(y_coords), np.max(x_coords), np.max(y_coords)]
            boxes.append(box)
            types.append('standard')

    # 2. Refined 10000 IDs (Crowd)
    if 10000 in mask:
        crowd_mask = (mask == 10000).astype(int)
        labeled_crowd, num_features = label(crowd_mask)
        for i in range(1, num_features + 1):
            y_c, x_c = np.where(labeled_crowd == i)
            if len(x_c) > 10: 
                box = [np.min(x_c), np.min(y_c), np.max(x_c), np.max(y_c)]
                boxes.append(box)
                types.append('refined')
                
    return boxes, types

"""-------------------------------------------------------------------------"""
if __name__ == "__main__":
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Image folder not found: {IMAGES_DIR}")
        
    # Get all available sequence directories
    all_img_seqs = sorted(os.listdir(IMAGES_DIR))[:1]
    all_mask_seqs = sorted(os.listdir(IMAGES_DIR))[:1]
    
    assert len(all_img_seqs) == len(all_mask_seqs), "Sequence has mismatched frame counts!"

    img_paths = []
    mask_paths = []
    for seq in all_img_seqs:
        seq_img_dir = os.path.join(IMAGES_DIR, seq)
        seq_mask_dir = os.path.join(MASKS_DIR, seq)
        
        images = sorted([f.name for f in Path(seq_img_dir).glob('*.png')])
        masks = sorted([f.name for f in Path(seq_mask_dir).glob('*.png')])

        for f_img, f_mask in zip(images, masks):
            img_paths.append(os.path.join(seq_img_dir, f_img))
            mask_paths.append(os.path.join(seq_mask_dir, f_mask))
            
    out_dir = OUT_DIR_GRID
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModel.from_pretrained("facebook/sam-vit-base").to(device)
    processor = AutoProcessor.from_pretrained("facebook/sam-vit-base")
    
    # Segment everything
    generator = pipeline("mask-generation", model="facebook/sam-vit-base", device=0)
    latencies = []
    seq = 0
    i = 0
    for img_p, mask_p in zip(img_paths, mask_paths):    
        raw_image = Image.open(img_p).convert("RGB")
        
        # --- START TIMING ---
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        points, p_types = get_points_with_metadata(mask_p)
        boxes, b_types = get_boxes_with_metadata(mask_p)
        
        if not points: continue
        if not boxes: continue
            
        """==========POINTS========================="""
        
        """
        # Inference
        input_points = [[ [p] for p in points ]]
        input_labels = [[ [1] for _ in points ]]
        
        inputs = processor(images=raw_image, input_points=input_points, 
                           input_labels=input_labels, return_tensors="pt").to(device)
        """
        
        """==========BOXES========================="""
        inputs = processor(images=raw_image, input_boxes=[boxes],
                           return_tensors="pt").to(device)        

        with torch.no_grad():
            outputs = model(**inputs)

        """
        outputs = generator(raw_image, points_per_batch=32)
        filename = os.path.basename(img_p)
        save_path = os.path.join(OUT_DIR_GRID, f"vis_{seq}_{filename}")
        
        visualize_everything(raw_image, outputs, save_path)
        """
        # Post-process
        # Note: masks[0] because we are processing one image at a time
        processed_masks = processor.post_process_masks(
            outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
        )[0] 

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.time() - start_time) * 1000
        latencies.append(elapsed_ms)
        """
        # Save visualization
        visualize_and_save(
            raw_image, 
            processed_masks,  
            points, 
            p_types,
            boxes,
            b_types,
            save_path
        )
        """
    if latencies:
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        
        print(f"--- Statistics over {len(latencies)} samples ---")
        print(f"Mean Inference Time: {mean_latency:.4f} ms")
        print(f"Std Dev:             {std_latency:.4f} ms")
        print(f"FPS:                 {1/mean_latency:.2f}")
            
        