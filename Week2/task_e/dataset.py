import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from pycocotools import mask as mask_utils

def collate_fn(batch):
    images, masks, boxes, classes = zip(*batch)
    return list(images), list(masks), list(boxes), list(classes)

class KITTIMOTSDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.img_root = os.path.join(root, "training/image_02")
        self.ann_root = os.path.join(root, "instances_txt")

        self.samples = []

        seqs = sorted(os.listdir(self.ann_root))

        for seq in seqs:
            ann_file = os.path.join(self.ann_root, seq)

            with open(ann_file) as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split(" ")

                frame = int(parts[0])
                track_id = int(parts[1])
                cls = int(parts[2])
                h = int(parts[3])
                w = int(parts[4])
                rle = parts[5]

                # ignore region
                if cls == 10:
                    continue

                img_path = os.path.join(
                    self.img_root,
                    seq.replace(".txt",""),
                    f"{frame:06d}.png"
                )

                self.samples.append({
                    "img": img_path,
                    "h": h,
                    "w": w,
                    "rle": rle,
                    "cls": cls,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = cv2.imread(sample["img"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        rle = {
            "counts": sample["rle"].encode("utf-8"),
            "size": [sample["h"], sample["w"]]
        }

        mask = mask_utils.decode(rle)
        ys, xs = np.where(mask > 0)
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        box = np.array([x1, y1, x2, y2])

        return image, mask.astype(np.float32), box, sample["cls"]