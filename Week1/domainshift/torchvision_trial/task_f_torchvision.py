import os
import copy
import random
import itertools
import numpy as np

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import utils
from engine import train_one_epoch, evaluate
from deart_dataset_torchvision import EuropeanArtDataset
from albumentations_aug import get_transforms


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


def get_model(num_classes: int, device: torch.device):
    # Load pretrained Faster R-CNN with ResNet-50-FPN backbone
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT"
    )

    # Replace classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes
    )

    return model.to(device)


def configure_trainable_params(model, unfreeze_mode: str):
    for p in model.parameters():
        p.requires_grad = False

    def _unfreeze(name):
        if unfreeze_mode == "all":
            return True

        if unfreeze_mode == "head_only":
            return "roi_heads" in name

        if unfreeze_mode == "head+rpn":
            return "roi_heads" in name or "rpn" in name

        if unfreeze_mode == "last_stage+head":
            return (
                "roi_heads" in name
                or "backbone.body.layer4" in name
            )

        raise ValueError(f"Unknown mode {unfreeze_mode}")

    trainable = 0
    total = 0

    for n, p in model.named_parameters():
        total += p.numel()
        if _unfreeze(n):
            p.requires_grad = True
            trainable += p.numel()

    print(
        f"Unfreeze mode: {unfreeze_mode} | "
        f"{trainable:,}/{total:,} ({100*trainable/total:.2f}%)"
    )

def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 1

    lr = 1e-4
    weight_decay = 0.01
    num_epochs = 5
    patience = 3
    min_delta = 0.0

    # Grid
    aug_options = [False, True]
    batch_sizes = [12]
    unfreeze_modes = ["head_only"]

    grid = list(itertools.product(aug_options, batch_sizes, unfreeze_modes))

    best_run = None

    for aug_enabled, batch_size, unfreeze_mode in grid:
        print("\n" + "=" * 90)
        print(
            f"RUN: aug={aug_enabled} | bs={batch_size} | lr={lr} | opt=AdamW | "
            f"unfreeze={unfreeze_mode} | epochs={num_epochs} | ES(pat={patience}, min_delta={min_delta})"
        )
        print("=" * 90)

        train_tf = get_transforms(enable=aug_enabled)
        val_tf = get_transforms(enable=False)

        train_set = EuropeanArtDataset(split="train", transforms=train_tf, debug=0.01)
        val_set = EuropeanArtDataset(split="val", transforms=val_tf, debug=0.01)

        num_classes = train_set.num_classes + 1

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

        model = get_model(
            num_classes=train_set.num_classes + 1,
            device=device
        )

        # Unfreeze strategy
        configure_trainable_params(model, unfreeze_mode)

        # Optimizer només amb params entrenables
        params = [p for p in model.parameters() if p.requires_grad]
        if len(params) == 0:
            print("WARNING: no trainable parameters! Skipping run.")
            continue

        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)
        run_best_map = -1.0
        run_best_state = None

        for epoch in range(num_epochs):
            train_one_epoch(
                model=model,
                optimizer=optimizer,
                data_loader=train_data_loader,
                device=device,
                epoch=epoch,
                print_freq=1,
            )

            coco_evaluator = evaluate(model, val_data_loader, device=device)
            current_map = float(coco_evaluator.coco_eval["bbox"].stats[0])

            print(
                f"[aug={aug_enabled} | bs={batch_size} | unfreeze={unfreeze_mode}] "
                f"Epoch {epoch+1}/{num_epochs} - val mAP: {current_map:.6f}"
            )

            if current_map > run_best_map:
                run_best_map = current_map
                run_best_state = copy.deepcopy(model.state_dict())

            early_stopper(current_map)
            if early_stopper.early_stop:
                print(f"Early stopping at epoch {epoch+1}. Best mAP: {run_best_map:.6f}")
                break

        # Guardar checkpoint per run
        run_ckpt = f"best_detr_aug{int(aug_enabled)}_bs{batch_size}_unfreeze{unfreeze_mode}_lr{lr}.pth"
        if run_best_state is not None:
            torch.save(
                {"model_state_dict": run_best_state, "best_map": run_best_map,
                 "cfg": {"aug": aug_enabled, "bs": batch_size, "lr": lr, "unfreeze": unfreeze_mode}},
                run_ckpt,
            )
            print(f"Saved best checkpoint -> {run_ckpt}")

        # Update best global
        if best_run is None or run_best_map > best_run["best_map"]:
            best_run = {"cfg": {"aug": aug_enabled, "bs": batch_size, "lr": lr, "unfreeze": unfreeze_mode},
                        "best_map": run_best_map, "ckpt": run_ckpt}

    print("\n" + "#" * 90)
    print("GRID SEARCH DONE")
    if best_run is None:
        print("No successful runs.")
        return
    print(
        f"Best config: aug={best_run['cfg']['aug']} | bs={best_run['cfg']['bs']} | "
        f"unfreeze={best_run['cfg']['unfreeze']} | lr={best_run['cfg']['lr']} | "
        f"best mAP={best_run['best_map']:.6f}"
    )
    print(f"Best checkpoint: {best_run['ckpt']}")
    print("#" * 90)


if __name__ == "__main__":
    main()