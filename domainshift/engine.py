import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from tqdm import tqdm
import numpy as np

def _to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        y = [_to_device(v, device) for v in x]
        return y if isinstance(x, list) else tuple(y)
    return x

def _build_hf_inputs(images, targets, processor, device):
    """
    Construeix inputs per DETR HF processor en format COCO.
    Espera targets amb keys: boxes (XYXY), labels, area (per objecte), iscrowd.
    """
    annotations = []

    for t in targets:
        # Boxes en XYXY -> XYWH (COCO)
        boxes_xyxy = t["boxes"]
        if torch.is_tensor(boxes_xyxy):
            boxes_xyxy = boxes_xyxy.detach().cpu()
        else:
            boxes_xyxy = torch.as_tensor(boxes_xyxy)

        if boxes_xyxy.numel() == 0:
            boxes_xywh = boxes_xyxy.reshape(0, 4)
        else:
            x1 = boxes_xyxy[:, 0]
            y1 = boxes_xyxy[:, 1]
            x2 = boxes_xyxy[:, 2]
            y2 = boxes_xyxy[:, 3]
            boxes_xywh = torch.stack([x1, y1, (x2 - x1), (y2 - y1)], dim=1)

        labels = t["labels"]
        if torch.is_tensor(labels):
            labels = labels.detach().cpu()
        else:
            labels = torch.as_tensor(labels)

        # Area: si no hi és, la calculem a partir de w*h
        if "area" in t:
            area = t["area"]
            if torch.is_tensor(area):
                area = area.detach().cpu()
            else:
                area = torch.as_tensor(area)
        else:
            if boxes_xywh.numel() == 0:
                area = torch.zeros((0,), dtype=torch.float32)
            else:
                area = (boxes_xywh[:, 2] * boxes_xywh[:, 3]).to(torch.float32)

        # iscrowd: si no hi és, tot a 0
        if "iscrowd" in t:
            iscrowd = t["iscrowd"]
            if torch.is_tensor(iscrowd):
                iscrowd = iscrowd.detach().cpu()
            else:
                iscrowd = torch.as_tensor(iscrowd)
        else:
            iscrowd = torch.zeros((labels.shape[0],), dtype=torch.int64)

        image_id = t.get("image_id", 0)
        if torch.is_tensor(image_id):
            image_id = int(image_id.item())

        objs = []
        for i in range(labels.shape[0]):
            objs.append(
                {
                    "bbox": boxes_xywh[i].tolist(),
                    "category_id": int(labels[i].item()),
                    "area": float(area[i].item()),
                    "iscrowd": int(iscrowd[i].item()),
                }
            )

        annotations.append({"image_id": image_id, "annotations": objs})

    # processor accepta torch tensors (C,H,W) o PIL/numpy
    inputs = processor(images=list(images), annotations=annotations, return_tensors="pt")

    # move to device
    inputs = _to_device(inputs, device)
    return inputs

def _force_hf_inputs_to_model_device(inputs, model):
    """Mou pixel_values / pixel_mask / labels al device del model (robust)."""
    model_device = next(model.parameters()).device

    if "pixel_values" in inputs and torch.is_tensor(inputs["pixel_values"]):
        inputs["pixel_values"] = inputs["pixel_values"].to(model_device, non_blocking=True)

    if "pixel_mask" in inputs and inputs["pixel_mask"] is not None and torch.is_tensor(inputs["pixel_mask"]):
        inputs["pixel_mask"] = inputs["pixel_mask"].to(model_device, non_blocking=True)

    # HF DETR: labels és List[Dict[str, Tensor]]
    if "labels" in inputs and inputs["labels"] is not None:
        new_labels = []
        for lab in inputs["labels"]:
            lab2 = {}
            for k, v in lab.items():
                lab2[k] = v.to(model_device, non_blocking=True) if torch.is_tensor(v) else v
            new_labels.append(lab2)
        inputs["labels"] = new_labels

    return inputs


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, processor, scaler=None):
    model.to(device)
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in tqdm(
            metric_logger.log_every(data_loader, print_freq, header),
            total=len(data_loader),
            desc=f"Epoch {epoch}",
    ):
        # build HF inputs (processor returns BatchFeature-like dict)
        inputs = _build_hf_inputs(images, targets, processor, device)

        inputs = _force_hf_inputs_to_model_device(inputs, model)

        # print("pixel_values:", inputs["pixel_values"].device, "| model:", next(model.parameters()).device)

        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=(scaler is not None)):
                outputs = model(**inputs)
                losses = outputs.loss
                loss_dict = outputs.loss_dict
        else:
            outputs = model(**inputs)
            losses = outputs.loss
            loss_dict = outputs.loss_dict

        loss_dict_reduced = utils.reduce_dict({k: v.detach() for k, v in loss_dict.items()})
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger
    
def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device, processor):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    # person_id = 1
    person_id = data_loader.dataset.label2id["person"]
    coco_evaluator.coco_eval["bbox"].params.catIds = [person_id]

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        target_sizes = torch.tensor([(img.shape[-2], img.shape[-1]) for img in images], device=device)

        inputs = _build_hf_inputs(images, targets, processor, device)

        inputs = _force_hf_inputs_to_model_device(inputs, model)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        model_time = time.time()
        outputs = model(**inputs)
        processed = processor.post_process_object_detection(outputs, target_sizes=target_sizes)
        model_time = time.time() - model_time

        # Adapt to coco evaluator expected format
        results = []
        for pred in processed:
            results.append({
                "boxes": pred["boxes"].to(cpu_device),
                "scores": pred["scores"].to(cpu_device),
                "labels": pred["labels"].to(cpu_device),
            })

        res = {target["image_id"]: output for target, output in zip(targets, results)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    # PER-CLASS METRICS 
    coco_eval = coco_evaluator.coco_eval['bbox']  # 'bbox' is your IoU type
    precision = coco_eval.eval['precision']       # shape: [IoU, recall, cat, area, maxDets]
    cat_ids = coco_eval.params.catIds             # category IDs evaluated

    print("\nPer-class mAP:")
    for idx, cat_id in enumerate(cat_ids):
        class_name = data_loader.dataset.id2label[cat_id]  # map ID → label
        mAP = np.mean(precision[:, :, idx, 0, -1])         # average over IoUs and recalls
        print(f"{class_name} (ID {cat_id}): mAP = {mAP:.4f}")
    # fi PER-CLASS METRICS 

    torch.set_num_threads(n_threads)
    return coco_evaluator
