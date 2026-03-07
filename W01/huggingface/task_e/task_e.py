import os
import copy
import random
import numpy as np

import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

import utils
from engine import train_one_epoch, evaluate
from task_e_dataset import KittiMotsDataset
from albumentations_aug import get_transforms

import time


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score: float):
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
            return
        if score >= self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def get_model(num_classes: int, device: torch.device, image_size: int):
    id2label = {0: "background", 1: "car", 2: "person"}
    label2id = {v: k for k, v in id2label.items()}

    processor = AutoImageProcessor.from_pretrained(
        "facebook/detr-resnet-50",
        size={"shortest_edge": image_size, "longest_edge": image_size},
    )

    model = AutoModelForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    return model, processor


def unfreeze_head_only(model):
    """
    Congela tot el model i només entrena els caps de classificació i bbox.
    """
    for p in model.parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if ("class_labels_classifier" in name) or ("bbox_predictor" in name):
            p.requires_grad = True


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3
    db_path = "/ghome/mcv/datasets/C5/KITTI-MOTS"
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"No existeix db_path: {db_path}")

    # Hiperparàmetres fixos
    lr = 1e-4
    weight_decay = 0.01
    num_epochs = 100
    patience = 3
    min_delta = 0.0

    # Config fixa
    aug_enabled = True
    batch_size = 16
    image_size = 960

    print("\n" + "=" * 90)
    print(
        f"RUN: size={image_size} | aug={aug_enabled} | bs={batch_size} | lr={lr} | "
        f"opt=AdamW | epochs={num_epochs} | ES(pat={patience}, min_delta={min_delta}) | unfreeze=head_only"
    )
    print("=" * 90)

    train_tf = get_transforms(enable=aug_enabled)
    val_tf = get_transforms(enable=False)

    train_set = KittiMotsDataset(db_path, train_tf, is_validation=False)
    val_set = KittiMotsDataset(db_path, val_tf, is_validation=True)

    train_data_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=utils.collate_fn,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        collate_fn=utils.collate_fn,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    model, processor = get_model(num_classes, device, image_size)
    unfreeze_head_only(model)

    # Optimizer només amb params entrenables
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        print("WARNING: no trainable parameters! Skipping run.")
        return

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)
    run_best_map = -1.0
    run_best_state = None
    run_time_total = 0.0

    last_avg_inf_time = None
    last_robustness = None

    for epoch in range(num_epochs):
        epoch_start = time.time()
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_data_loader,
            device=device,
            epoch=epoch,
            print_freq=100,
            processor=processor,
        )
        run_time_total += time.time() - epoch_start

        eval_start = time.time()
        coco_evaluator, avg_inf_time = evaluate(model, val_data_loader, device=device, processor=processor)
        eval_time = time.time() - eval_start
        run_time_total += eval_time

        current_map = float(coco_evaluator.coco_eval["bbox"].stats[0])
        robustness_mar = float(coco_evaluator.coco_eval["bbox"].stats[8])  # mAR@100
        trainable_params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        total_params_m = sum(p.numel() for p in model.parameters()) / 1e6
        last_avg_inf_time = avg_inf_time
        last_robustness = robustness_mar

        print(
            f"[size={image_size} | aug={aug_enabled} | bs={batch_size} | unfreeze=head_only] "
            f"Epoch {epoch+1}/{num_epochs} - mAP: {current_map:.6f} | "
            f"robustness(mAR@100): {robustness_mar:.6f} | "
            f"trainable params: {trainable_params_m:.2f}M / total {total_params_m:.2f}M | "
            f"avg inference: {avg_inf_time:.4f}s/batch | eval time: {eval_time:.1f}s"
        )

        if current_map > run_best_map:
            run_best_map = current_map
            run_best_state = copy.deepcopy(model.state_dict())

        early_stopper(current_map)
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch+1}. Best mAP: {run_best_map:.6f}")
            break

    # Guardar checkpoint
    run_ckpt = f"best_detr_size{image_size}_headonly_aug{int(aug_enabled)}_bs{batch_size}_lr{lr}.pth"
    if run_best_state is not None:
        torch.save(
            {
                "model_state_dict": run_best_state,
                "best_map": run_best_map,
                "cfg": {"size": image_size, "aug": aug_enabled, "bs": batch_size, "lr": lr, "unfreeze": "head_only"},
            },
            run_ckpt,
        )
        print(f"Saved best checkpoint -> {run_ckpt}")
        print(
            f"Run summary -> best mAP: {run_best_map:.6f} | "
            f"last robustness(mAR@100): {last_robustness:.6f} | "
            f"total train+eval time: {run_time_total/60:.1f} min | "
            f"avg inference (last eval): {last_avg_inf_time:.4f}s/batch"
        )
    else:
        print("Run finished without a valid checkpoint.")


if __name__ == "__main__":
    main()
