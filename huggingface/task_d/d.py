import time
import os
import matplotlib.pyplot as plt

import torch
import torchvision.transforms.v2 as F
from transformers import DetrForObjectDetection, DetrImageProcessor

import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

from task_d_dataset import KittiMotsDataset


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _mean_precision_over_classes(prec):
    """
    COCO precision array shape: [T, R, K, A, M]
    Returns recall (len R) and mean precision over IoU thresholds, classes, areas, maxDets.
    """
    # Average over IoU thresholds, classes, areas, maxDets
    pr = prec.mean(axis=(0, 2, 3, 4))
    return pr


def plot_pr_curve(coco_eval, out_path):
    prec = coco_eval.eval["precision"]  # [T, R, K, A, M]
    recalls = coco_eval.params.recThrs   # len R
    pr = _mean_precision_over_classes(prec)

    plt.figure(figsize=(6, 4))
    plt.plot(recalls, pr, label="mAP@0.5:0.95")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (mean over classes)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_map_per_class(coco_eval, out_path):
    """
    Compute per-class mAP@0.5:0.95 and plot bar chart.
    """
    prec = coco_eval.eval["precision"]  # [T, R, K, A, M]

    # Robust category retrieval
    cat_ids = coco_eval.params.catIds
    if cat_ids is None:
        cat_ids = list(coco_eval.cocoGt.cats.keys())

    cat_names = []
    for cid in cat_ids:
        cats = coco_eval.cocoGt.loadCats(cid)
        if cats:
            cat_names.append(cats[0].get("name", str(cid)))
        else:
            cat_names.append(str(cid))

    ap_per_class = []
    for k in range(len(cat_ids)):
        p = prec[:, :, k, 0, -1]  # all IoUs, all recalls, area=all, maxDet=last
        p = p[p > -1]
        ap = p.mean() if p.size else float("nan")
        ap_per_class.append(ap)

    plt.figure(figsize=(8, 4))
    plt.bar(cat_names, ap_per_class, color="#4e79a7")
    plt.ylabel("mAP @ 0.5:0.95")
    plt.title("mAP by class")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_map_vs_params(mAP, num_params_m, out_path):
    plt.figure(figsize=(5, 4))
    plt.scatter([num_params_m], [mAP], color="#e15759", s=80)
    plt.xlabel("Parameters (millions)")
    plt.ylabel("mAP @ 0.5:0.95")
    plt.title("mAP vs Parameters")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def box_iou(box1, box2):
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


def accumulate_pr_curve(pred_store, gt_by_image, total_gt, iou_thr=0.5):
    """
    pred_store: list of dict(score, box (tensor), label, image_id)
    gt_by_image: dict[image_id] -> list of (box tensor, label int)
    Returns thresholds, precisions, recalls, f1s
    """
    pred_store = sorted(pred_store, key=lambda x: x["score"], reverse=True)
    tp_cum, fp_cum = 0, 0
    matched = {}  # (image_id, label) -> set(idx)
    precisions, recalls, f1s, confs = [], [], [], []

    for pred in pred_store:
        key = (pred["image_id"], pred["label"])
        used = matched.setdefault(key, set())
        gt_items = gt_by_image.get(pred["image_id"], [])

        best_iou, best_idx = 0.0, -1
        for idx, (g_box, g_label) in enumerate(gt_items):
            if g_label != pred["label"] or idx in used:
                continue
            iou = box_iou(pred["box"], g_box)
            if iou > best_iou:
                best_iou, best_idx = iou, idx

        if best_iou >= iou_thr:
            tp_cum += 1
            used.add(best_idx)
        else:
            fp_cum += 1

        precision = tp_cum / (tp_cum + fp_cum + 1e-6)
        recall = tp_cum / (total_gt + 1e-6)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        confs.append(pred["score"])

    return confs, precisions, recalls, f1s


def plot_f1_confidence(confs, f1s, out_path):
    plt.figure(figsize=(6, 4))
    plt.plot(confs, f1s, color="#59a14f")
    plt.xlabel("Confidence threshold")
    plt.ylabel("F1 score")
    plt.title("F1 vs confidence")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


@torch.inference_mode()
def evaluate(model, processor, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    coco_evaluator = CocoEvaluator(coco, ["bbox"])
    pred_store = []
    gt_by_image = {}
    total_gt = 0

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        # DeTR espera PIL o array; convertim des de tensors normalitzats (0-1) a uint8
        pil_images = [
            F.functional.to_pil_image((img * 255).clamp(0, 255).to(torch.uint8).cpu())
            for img in images
        ]

        inputs = processor(images=pil_images, return_tensors="pt").to(device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(**inputs)
        model_time = time.time() - model_time

        target_sizes = torch.tensor([img.shape[-2:] for img in images], device=device)
        processed = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.01
        )

        detections = [
            {
                "boxes": det["boxes"].to(device),
                "scores": det["scores"].to(device),
                "labels": det["labels"].to(device),
            }
            for det in processed
        ]

        res = {target["image_id"]: output for target, output in zip(targets, detections)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        # Store predictions and GT for F1-confidence curve (IoU=0.5)
        for target, det in zip(targets, processed):
            image_id = target["image_id"]
            if torch.is_tensor(image_id):
                image_id = int(image_id.item())
            else:
                image_id = int(image_id)

            g_boxes = target["boxes"].cpu()
            g_labels = target["labels"].cpu()
            total_gt += len(g_boxes)
            gt_by_image.setdefault(image_id, [])
            for g_box, g_lab in zip(g_boxes, g_labels):
                gt_by_image[image_id].append((g_box, int(g_lab)))

            for b, s, l in zip(det["boxes"].cpu(), det["scores"].cpu(), det["labels"].cpu()):
                pred_store.append({
                    "score": float(s),
                    "box": b,
                    "label": int(l),
                    "image_id": image_id,
                })

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    
    return coco_evaluator, pred_store, gt_by_image, total_gt

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transformation = F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=None),
    ])
    
    db_path = "/ghome/mcv/datasets/C5/KITTI-MOTS"
    dataset = KittiMotsDataset(db_path, transformation)
    
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=utils.collate_fn
    )
    
    # Initialize pretrained DeTR (COCO) from Hugging Face
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
    
    coco_eval, pred_store, gt_by_image, total_gt = evaluate(model, processor, data_loader, device=device)

    # Save plots
    plots_dir = "/ghome/group07/MCV-C5-Group7/huggingface/task_d/plots"
    _ensure_dir(plots_dir)
    plot_pr_curve(coco_eval.coco_eval["bbox"], os.path.join(plots_dir, "pr_curve.png"))
    plot_map_per_class(coco_eval.coco_eval["bbox"], os.path.join(plots_dir, "map_per_class.png"))

    # F1-confidence curve (IoU=0.5)
    if total_gt > 0 and pred_store:
        confs, _, _, f1s = accumulate_pr_curve(pred_store, gt_by_image, total_gt, iou_thr=0.5)
        plot_f1_confidence(confs, f1s, os.path.join(plots_dir, "f1_confidence.png"))

    # mAP vs parameters (single point for current model)
    mAP = float(coco_eval.coco_eval["bbox"].stats[0])  # mAP@0.5:0.95
    num_params_m = sum(p.numel() for p in model.parameters()) / 1e6
    plot_map_vs_params(mAP, num_params_m, os.path.join(plots_dir, "map_vs_params.png"))
