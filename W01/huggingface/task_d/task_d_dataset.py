import os
from pathlib import Path

import torch

from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
import torchvision.transforms.v2 as F

class KittiMotsDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms

        self.img_paths = []
        self.mask_paths = []

        # Get all sequence directories (e.g., '0000', '0001')
        img_root = os.path.join(root, "training/image_02")
        mask_root = os.path.join(root, "instances")
        
        # Sort sequences to maintain consistency
        img_seqs = sorted(os.listdir(img_root))
        mask_seqs = sorted(os.listdir(mask_root))
        
        assert img_seqs == mask_seqs, "Mismatch between image and mask sequence folders!"
        
        for seq in img_seqs:
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
        mask = read_image(mask_path).to(dtype=torch.int32)
        
        # instances are encoded as different colors
        obj_ids = torch.unique(mask).to(dtype=torch.int32)
        # filter for values between 1 and 9999 (removes background and ignore regions)
        valid_ids = obj_ids[(obj_ids > 0) & (obj_ids < 10000)].to(dtype=torch.int32)
        num_objs = len(valid_ids)
        
        # split the color-encoded mask into a set
        # of binary masks
        masks = (mask == valid_ids[:, None, None]).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        kitti_classes = valid_ids // 1000
        labels = torch.zeros_like(kitti_classes)
        # If it was a 1 (Car), make it a 3 (COCO Car)
        labels[kitti_classes == 1] = 3
        # If it was a 2 (Pedestrian), make it a 1 (COCO Person)
        labels[kitti_classes == 2] = 1

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.functional.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img / 255, target

    def __len__(self):
        return len(self.img_paths)
