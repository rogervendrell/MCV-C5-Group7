import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
import torchvision.transforms.v2 as F

class KittiMotsDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, is_validation):
        self.root = root
        self.transforms = transforms

        self.img_paths = []
        self.mask_paths = []
        
        img_root = os.path.join(root, "training/image_02")
        mask_root = os.path.join(root, "instances")

        # Define your validation sequence indices
        val_indices = [2, 6, 7, 8, 10, 13, 14, 16, 18]
        
        # Get all available sequence directories
        all_img_seqs = sorted(os.listdir(img_root))
        all_mask_seqs = sorted(os.listdir(mask_root))
        
        # Filter based on the train boolean
        if is_validation:
            # Get only the sequences at the specified indices
            self.img_seqs = [all_img_seqs[i] for i in val_indices]
            self.mask_seqs = [all_mask_seqs[i] for i in val_indices]
        else:
            # Get everything EXCEPT those indices
            self.img_seqs = [s for i, s in enumerate(all_img_seqs) if i not in val_indices]
            self.mask_seqs = [s for i, s in enumerate(all_mask_seqs) if i not in val_indices]
        
        assert self.img_seqs == self.mask_seqs, "Mismatch between image and mask sequence folders!"
        
        for seq in self.img_seqs:
            seq_img_dir = os.path.join(img_root, seq)
            seq_mask_dir = os.path.join(mask_root, seq)
            
            # Ensure it's actually a directory
            if not os.path.isdir(seq_img_dir):
                continue
            if not os.path.isdir(seq_mask_dir):
                continue
            
            # 2. Sort the frames within each sequence
            frames = sorted([f.name for f in Path(seq_img_dir).glob('*.png')])
            mask_frames = sorted([f.name for f in Path(seq_mask_dir).glob('*.png')])

            assert len(frames) == len(mask_frames), f"Sequence {seq} has mismatched frame counts!"
            
            for f_img, f_mask in zip(frames, mask_frames):
                self.img_paths.append(os.path.join(seq, f_img))
                self.mask_paths.append(os.path.join(seq, f_mask))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "training/image_02", self.img_paths[idx])
        mask_path = os.path.join(self.root, "instances", self.mask_paths[idx])
        
        img = read_image(img_path).to(dtype=torch.int32)

        # KITTI-MOTS instance masks are single-channel where value = class*1000 + instance_id
        mask_np = np.array(Image.open(mask_path), dtype=np.int32)
        mask = torch.from_numpy(mask_np)

        obj_ids = torch.unique(mask).to(dtype=torch.int32)
        # filter for values between 1 and 9999 (removes background and ignore regions)
        valid_ids = obj_ids[(obj_ids > 0) & (obj_ids < 10000)].to(dtype=torch.int32)

        # binary masks per instance: shape (N, H, W)
        masks = (mask[None, ...] == valid_ids[:, None, None]).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        labels = valid_ids // 1000

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        keep = area > 0
        boxes = boxes[keep]
        labels = labels[keep]
        masks = masks[keep]
        area = area[keep]
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        
        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.functional.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels.to(dtype=torch.int64)
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # Keep original intensity range; processor will handle normalization/resizing
        return img, target

    def __len__(self):
        return len(self.img_paths)
