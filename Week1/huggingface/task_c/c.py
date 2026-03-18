import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor

# Paths to the validation split (update if you change your dataset layout)
IMAGES_DIR = Path("/ghome/group07/MCV-C5-Group7/ultralytics/dataset/images/val")
LABELS_DIR = Path("/ghome/group07/MCV-C5-Group7/ultralytics/dataset/labels/val")
OUT_DIR = Path("/ghome/group07/MCV-C5-Group7/huggingface/bboxes")

# YOLO label id -> human-readable name (adapt to your classes)
CLASS_MAP = {0: "person", 2: "car"}

# Inference config
MODEL_NAME = "facebook/detr-resnet-50"
SCORE_THR = 0.25  # keep low; TP/FP/FN separation will still filter by IoU
NUM_RANDOM_IMAGES = 5  # how many images to sample from the split


def stitch_detection_quadrants(image, boxes):
    """
    Build a 2x2 grid with GT / TP / FP / FN boxes drawn separately.
    Expects `image` as HxWx3 uint8 in BGR order (OpenCV friendly)
    and `boxes` as a list of {"bbox": (x1, y1, x2, y2), "label": str}.
    """

    # Create 4 copies of the image
    img_gt = image.copy()
    img_tp = image.copy()
    img_fp = image.copy()
    img_fn = image.copy()

    # Color scheme (BGR for OpenCV)
    colors = {
        "gt": (0, 255, 0),      # Green
        "tp": (255, 0, 0),      # Blue
        "fp": (0, 0, 255),      # Red
        "fn": (0, 255, 255),    # Yellow
    }

    # Draw boxes on corresponding quadrant image
    for item in boxes:
        (x1, y1, x2, y2) = item["bbox"]
        label = item["label"]

        if label == "gt":
            cv2.rectangle(img_gt, (x1, y1), (x2, y2), colors["gt"], 2)

        elif label == "tp":
            cv2.rectangle(img_tp, (x1, y1), (x2, y2), colors["tp"], 2)

        elif label == "fp":
            cv2.rectangle(img_fp, (x1, y1), (x2, y2), colors["fp"], 2)

        elif label == "fn":
            cv2.rectangle(img_fn, (x1, y1), (x2, y2), colors["fn"], 2)

    # Stack images
    top_row = np.hstack((img_gt, img_tp))
    bottom_row = np.hstack((img_fp, img_fn))
    stitched = np.vstack((top_row, bottom_row))

    return stitched


def load_yolo_labels(label_path, img_w, img_h):
    """Parse a YOLO-format .txt file into pixel xyxy boxes and class names."""
    if not label_path.exists():
        return [], []

    boxes, labels = [], []
    with label_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue  # skip malformed rows
            cls_id, xc, yc, w, h = map(float, parts)

            # Convert from normalized center/size to absolute xyxy
            xc *= img_w
            yc *= img_h
            w *= img_w
            h *= img_h

            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2

            box = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
            boxes.append(box)
            labels.append(CLASS_MAP.get(int(cls_id), str(int(cls_id))))

    return boxes, labels


def box_iou(box1, box2):
    """IoU for two boxes in xyxy format."""
    inter_x1 = torch.maximum(box1[0], box2[0])
    inter_y1 = torch.maximum(box1[1], box2[1])
    inter_x2 = torch.minimum(box1[2], box2[2])
    inter_y2 = torch.minimum(box1[3], box2[3])

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter_area + 1e-6
    return inter_area / union


def classify_detections(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thr=0.5):
    """Greedy match with class check to tag preds as TP/FP and find FN gts."""
    gt_used = [False] * len(gt_boxes)
    tp_idx, fp_idx, fn_idx = [], [], []

    for i, p_box in enumerate(pred_boxes):
        best_iou, best_j = 0.0, -1
        for j, (g_box, g_lab) in enumerate(zip(gt_boxes, gt_labels)):
            if gt_used[j]:
                continue
            if pred_labels[i] != g_lab:
                continue
            iou = box_iou(p_box, g_box)
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_iou >= iou_thr:
            tp_idx.append((i, best_j))
            gt_used[best_j] = True
        else:
            fp_idx.append(i)

    for j, used in enumerate(gt_used):
        if not used:
            fn_idx.append(j)

    return tp_idx, fp_idx, fn_idx


if __name__ == "__main__":
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Image folder not found: {IMAGES_DIR}")

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = DetrImageProcessor.from_pretrained(MODEL_NAME)
    model = DetrForObjectDetection.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    all_images = [
        p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not all_images:
        raise FileNotFoundError(f"No images found in {IMAGES_DIR}")

    image_paths = random.sample(
        all_images, k=min(NUM_RANDOM_IMAGES, len(all_images))
    )

    for idx, img_path in enumerate(image_paths, start=1):
        label_path = LABELS_DIR / (img_path.stem + ".txt")

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[skip] Unable to read image: {img_path}")
            continue

        # Prepare GT
        h, w = img_bgr.shape[:2]
        gt_boxes, gt_labels = load_yolo_labels(label_path, w, h)

        # Prepare image for model (RGB array)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            inputs = processor(images=img_rgb, return_tensors="pt").to(device)
            outputs = model(**inputs)

        target_sizes = torch.tensor([[h, w]], device=device)
        result = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=SCORE_THR
        )[0]

        pred_boxes = result["boxes"].cpu()
        pred_scores = result["scores"].cpu()
        pred_labels_ids = result["labels"].cpu()
        id2label = model.config.id2label
        pred_labels = [id2label[int(i)].lower() for i in pred_labels_ids]

        # Classify detections vs GT
        tp_idx, fp_idx, fn_idx = classify_detections(pred_boxes, pred_labels, gt_boxes, gt_labels)

        boxes_for_plot = []

        # GT boxes
        for g_box in gt_boxes:
            boxes_for_plot.append({
                "bbox": tuple(g_box.to(torch.int64).tolist()),
                "label": "gt",
            })

        # TP boxes
        for p_idx, _ in tp_idx:
            boxes_for_plot.append({
                "bbox": tuple(pred_boxes[p_idx].to(torch.int64).tolist()),
                "label": "tp",
            })

        # FP boxes
        for p_idx in fp_idx:
            boxes_for_plot.append({
                "bbox": tuple(pred_boxes[p_idx].to(torch.int64).tolist()),
                "label": "fp",
            })

        # FN boxes
        for g_idx in fn_idx:
            boxes_for_plot.append({
                "bbox": tuple(gt_boxes[g_idx].to(torch.int64).tolist()),
                "label": "fn",
            })

        stitched = stitch_detection_quadrants(img_bgr, boxes_for_plot)

        # Replace the leading 4-digit prefix, keep the rest of the name
        stem = img_path.stem
        if len(stem) >= 4 and stem[:4].isdigit():
            new_stem = f"{idx:04d}" + stem[4:]
        else:
            new_stem = f"{idx:04d}_{stem}"

        out_path = out_dir / f"{new_stem}_quad.png"
        cv2.imwrite(str(out_path), stitched)
        print(
            f"[{idx}/{len(image_paths)}] Saved {out_path} | "
            f"GT {len(gt_boxes)} | TP {len(tp_idx)} | FP {len(fp_idx)} | FN {len(fn_idx)}"
        )
