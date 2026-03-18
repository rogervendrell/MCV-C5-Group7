from typing import Any, List, Dict, Optional, Union, Tuple

import os
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
import time
from Utils import get_boxes, refine_masks, load_image
#from plot_utils import plot_detections, plot_detections_plotly
from result_utils import DetectionResult
import pycocotools.mask as mask_utils


# Paths to the validation split (update if you change your dataset layout)
IMAGES_DIR = Path("/ghome/mcv/datasets/C5/KITTI-MOTS/training/image_02/")
OUT_DIR = Path("/ghome/group07/MCV-C5-Group7/Week2/task_b/predictions/")

"""-------------------------------------------------------------------------"""
def detect(
    detector,
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """

    labels = [label if label.endswith(".") else label+"." for label in labels]

    results = detector(image,  candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results

"""-------------------------------------------------------------------------"""
def segment(
    segmentator,
    processor,
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    polygon_refinement: bool = False,
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    boxes = get_boxes(detection_results)
    if not boxes[0]:
        return None
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)
     
     
    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results

"""-------------------------------------------------------------------------"""
def grounded_segmentation(
    detector,
    segmentator,
    processor,
    image: Union[Image.Image, str],
    labels: List[str],
    threshold: float = 0.3,
    polygon_refinement: bool = False,
) -> Tuple[np.ndarray, List[DetectionResult]]:
    if isinstance(image, str):
        image = load_image(image)

    detections = detect(detector, image, labels, threshold)
    detections = segment(segmentator, processor, image, detections, polygon_refinement)

    return np.array(image), detections

"""-------------------------------------------------------------------------"""
def label_to_class(label):
    if label == "a car.": return 1
    if label == "a person.": return 2

"""-------------------------------------------------------------------------"""
if __name__ == "__main__":
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Image folder not found: {IMAGES_DIR}")
        
    # Get all available sequence directories
    all_img_seqs = sorted(os.listdir(IMAGES_DIR))[:1]

    img_paths = []

    for seq in all_img_seqs:
        seq_img_dir = IMAGES_DIR / seq
        images = sorted(seq_img_dir.glob("*.png"))
        img_paths.extend(images)
            
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    detector = pipeline(model="IDEA-Research/grounding-dino-tiny", task="zero-shot-object-detection", device=device)
    segmentator = AutoModelForMaskGeneration.from_pretrained("facebook/sam-vit-base").to(device)
    processor = AutoProcessor.from_pretrained("facebook/sam-vit-base")
    
    seq = 0
    i = 0
    coco_results = []
    latencies = []
    with torch.no_grad():
      for img_p in img_paths:
        labels = ["a car.", "a person."]
        threshold = 0.1
          
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
                
        detector_id = "IDEA-Research/grounding-dino-tiny"
        segmenter_id = "facebook/sam-vit-base"
        
        image_array, detections = grounded_segmentation(
            detector,
            segmentator,
            processor,
            image=img_p.__str__(),
            labels=labels,
            threshold=threshold,
            polygon_refinement=True,
        )
        if detections == None:
            continue
          
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.time() - start_time) * 1000
        latencies.append(elapsed_ms)
                
        """
        #filename = os.path.basename(img_p)
        #save_path = os.path.join(OUT_DIR, f"vis_{seq}_{filename}")
        #plot_detections(image_array, detections, save_path)
        
        if i == 1:
            seq = seq + 1
            i = 0
        else:
            i = i + 1

        seq = int(img_p.parent.name)
        frame = int(img_p.stem)
        img_id = seq * 100000 + frame

        for res in detections:
            box = res.box.xyxy
            category_id = label_to_class(res.label)
            score = res.score
            mask = res.mask

            mask = (mask > 0).astype(np.uint8)
            rle = mask_utils.encode(np.asfortranarray(mask))
            rle["counts"] = rle["counts"].decode("utf-8")
            
            coco_results.append({
                "image_id": img_id,
                "category_id": category_id,
                "segmentation": rle,
                "score": score
            })
    # SAVE PREDICTIONS
    pred_file = out_dir / "groundedsam_predictions_005.json"

    with open(pred_file, "w") as f:
        json.dump(coco_results, f)
    """
    if latencies:
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        
        print(f"--- Statistics over {len(latencies)} samples ---")
        print(f"Mean Inference Time: {mean_latency:.4f} ms")
        print(f"Std Dev:             {std_latency:.4f} ms")
        print(f"FPS:                 {1/mean_latency:.2f}")