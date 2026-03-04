import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from engine import evaluate
from deart_dataset import EuropeanArtDataset
import utils


def load_model(num_classes, id2label, label2id, device, checkpoint_path=None):
    processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")

    model = AutoModelForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    if checkpoint_path is not None:
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print("Checkpoint loaded successfully.")

    return model, processor


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validation dataset
    val_set = EuropeanArtDataset(split="val", transforms=None)

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        collate_fn=utils.collate_fn,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    # print("\n==============================")
    # print("Evaluating PRETRAINED DETR")
    # print("==============================")

    # processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")

    # model_pretrained = AutoModelForObjectDetection.from_pretrained(
    #     "facebook/detr-resnet-50",
    #     ignore_mismatched_sizes=True,
    # ).to(device)

    # coco_eval_pre = evaluate(model_pretrained, val_loader, device, processor)
    # map_pre = float(coco_eval_pre.coco_eval["bbox"].stats[0])
    # print(f"\nPretrained mAP: {map_pre:.6f}")

    print("\n==============================")
    print("Evaluating FINETUNED DETR (deart)")
    print("==============================")

    model_finetuned, processor = load_model(
        num_classes=val_set.num_classes,
        id2label=val_set.id2label,
        label2id=val_set.label2id,
        device=device,
        checkpoint_path="/ghome/group07/MCV-C5-Group7/domainshift/copia/checkpoint_epoch4.pth",
    )

    coco_eval_fine = evaluate(model_finetuned, val_loader, device, processor)
    map_fine = float(coco_eval_fine.coco_eval["bbox"].stats[0])
    print(f"\nFinetuned mAP: {map_fine:.6f}")



    print("\n==============================")
    print("Evaluating FINETUNED DETR (kitti)")
    print("==============================")
    num_classes = 3
    id2label = {0: "background", 1: "car", 2: "person"}
    label2id = {v: k for k, v in id2label.items()}

    model_finetuned, processor = load_model(
        num_classes=num_classes,
        id2label=id2label,
        label2id=label2id,
        device=device,
        checkpoint_path="/export/home/group07/MCV-C5-Group7/huggingface/runs/train_d_group07_105049/best_detr_size960_headonly_aug1_bs16_lr0.0001.pth",
    )

    coco_eval_fine = evaluate(model_finetuned, val_loader, device, processor)
    map_fine = float(coco_eval_fine.coco_eval["bbox"].stats[0])
    print(f"\nFinetuned mAP: {map_fine:.6f}")

    print("\n=====================================")
    print("FINAL COMPARISON")
    print("=====================================")
    print(f"Pretrained mAP : {map_pre:.6f}")
    print(f"Finetuned mAP  : {map_fine:.6f}")
    print(f"Improvement    : {map_fine - map_pre:.6f}")


if __name__ == "__main__":
    main()