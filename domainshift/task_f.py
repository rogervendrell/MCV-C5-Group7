import os
import copy
import random
import itertools
import numpy as np

import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

import utils
from engine import train_one_epoch, evaluate
from deart_dataset import EuropeanArtDataset
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


def get_model(num_classes: int, id2label, label2id, device: torch.device):
    processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")

    model = AutoModelForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    return model, processor


def configure_trainable_params(model, unfreeze_mode: str):
    """
    Controla quines parts entrenes.
    Modes:
      - all: tot entrenable
      - head_only: només class + bbox heads
      - head+transformer: head + transformer encoder/decoder (backbone congelat)
      - last_stage+head: últim stage backbone + head (transformer congelat)
    """
    # Primer, congelem tot
    for p in model.parameters():
        p.requires_grad = False

    def _unfreeze_if_name_matches(name: str) -> bool:
        if unfreeze_mode == "all":
            return True

        # HF DETR: cap de detecció (class/bbox)
        if unfreeze_mode == "head_only":
            return ("class_labels_classifier" in name) or ("bbox_predictor" in name)

        # head + transformer (encoder/decoder)
        if unfreeze_mode == "head+transformer":
            if ("class_labels_classifier" in name) or ("bbox_predictor" in name):
                return True
            if ("model.encoder" in name) or ("model.decoder" in name):
                return True
            return False

        # últim stage backbone + head
        if unfreeze_mode == "last_stage+head":
            if ("class_labels_classifier" in name) or ("bbox_predictor" in name):
                return True
            # backbone sol ser: model.backbone.model (timm/resnet)
            # últim stage típic: layer4.*
            if ("model.backbone" in name) and ("layer4" in name):
                return True
            return False

        raise ValueError(f"Unknown unfreeze_mode: {unfreeze_mode}")

    # Descongelem segons mode
    trainable = 0
    total = 0
    for n, p in model.named_parameters():
        total += p.numel()
        if _unfreeze_if_name_matches(n):
            p.requires_grad = True
            trainable += p.numel()

    print(f"Unfreeze mode: {unfreeze_mode} | trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


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
    batch_sizes = [8]
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

        train_tf = get_transforms(enable=False)
        val_tf = get_transforms(enable=False)

        train_set = EuropeanArtDataset(split="train", transforms=train_tf)
        val_set = EuropeanArtDataset(split="val", transforms=val_tf)

        num_classes = train_set.num_classes

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

        model, processor = get_model(
            num_classes=train_set.num_classes,
            id2label=train_set.id2label,
            label2id=train_set.label2id,
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
                print_freq=100,
                processor=processor,
            )

            # coco_evaluator = evaluate(model, val_data_loader, device=device, processor=processor)
            # current_map = float(coco_evaluator.coco_eval["bbox"].stats[0])

            print(
                f"[aug={aug_enabled} | bs={batch_size} | unfreeze={unfreeze_mode}] "
                f"Epoch {epoch+1}/{num_epochs}" # - val mAP: {current_map:.6f}"
            )

            path = f"checkpoint_epoch{epoch}.pth"
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch,
                 "cfg": {"aug": aug_enabled, "bs": batch_size, "lr": lr, "unfreeze": unfreeze_mode}},
                path,
            )
            print(f"Saved checkpoint -> {path}")

            # if current_map > run_best_map:
            #     run_best_map = current_map
            #     run_best_state = copy.deepcopy(model.state_dict())

            # early_stopper(current_map)
            # if early_stopper.early_stop:
            #     print(f"Early stopping at epoch {epoch+1}. Best mAP: {run_best_map:.6f}")
            #     break

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