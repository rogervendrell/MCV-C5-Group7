import torch
import json
from datasets import load_dataset, load_from_disk
from torchvision import tv_tensors
import torchvision.transforms.v2 as F
from torch.utils.data import random_split


class EuropeanArtDataset(torch.utils.data.Dataset):
    def __init__(self, split="train", transforms=None, val_ratio=0.1, seed=42, debug=None):
        full_ds = load_from_disk("/ghome/group07/MCV-C5-Group7/domainshift/dataset")

        # Build global mapping once
        all_names = set()
        for sample in full_ds:
            ann = json.loads(sample["annotations"])
            for cat in ann["categories"]:
                all_names.add(cat["name"])

        all_names = sorted(list(all_names))
        self.label2id = {name: i for i, name in enumerate(all_names)}
        self.id2label = {i: name for name, i in self.label2id.items()}
        self.num_classes = len(self.label2id)

        torch.manual_seed(seed)
        val_size = int(len(full_ds) * val_ratio)
        train_size = len(full_ds) - val_size
        train_ds, val_ds = random_split(full_ds, [train_size, val_size])

        self.ds = train_ds if split == "train" else val_ds

        if debug is not None:
            debug_size = max(1, int(len(self.ds) * debug))
            indices = torch.randperm(len(self.ds))[:debug_size]
            self.ds = torch.utils.data.Subset(self.ds, indices)
            print(f"[DEBUG MODE] Using {debug_size}/{len(indices)} samples ({debug*100:.2f}%)")

        self.to_image = F.ToImage()
        self.transforms = transforms

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        try:
            sample = self.ds[idx]
            img = self.to_image(sample["image"])

            ann_dict = json.loads(sample["annotations"])
            objects = ann_dict["annotations"]

            local_id_to_name = {
                c["id"]: c["name"]
                for c in ann_dict["categories"]
            }

            boxes, labels, areas, iscrowd = [], [], [], []

            for obj in objects:
                category_name = local_id_to_name[obj["category_id"]]
                if category_name != "person":
                    continue

                x, y, w, h = obj["bbox"]
                boxes.append([x, y, x + w, y + h])

                global_id = self.label2id[category_name]
                labels.append(global_id)
                areas.append(obj["area"])
                iscrowd.append(obj["iscrowd"])

            if len(boxes) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
                areas = torch.zeros((0,), dtype=torch.float32)
                iscrowd = torch.zeros((0,), dtype=torch.int64)
            else:
                boxes = torch.tensor(boxes, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.int64)
                areas = torch.tensor(areas, dtype=torch.float32)
                iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

            target = {
                "boxes": tv_tensors.BoundingBoxes(
                    boxes,
                    format="XYXY",
                    canvas_size=img.shape[-2:],
                ),
                "labels": labels,
                "area": areas,
                "iscrowd": iscrowd,
                "image_id": idx,
            }

            if self.transforms:
                img, target = self.transforms(img, target)

            return img, target

        except Exception as e:
            print(f"[WARNING] Skipping corrupted sample at idx {idx}: {e}")
            return None