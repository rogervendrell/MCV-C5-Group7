import os
from pathlib import Path
import torch
import numpy as np
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
import torchvision.transforms.v2 as F

class KittiMotsDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, bTransforms, bHandleIgnore, is_validation):
        self.root = root
        self.transforms = transforms
        self.bTransforms = bTransforms
        self.bHandleIgnore = bHandleIgnore

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
        
        img = read_image(img_path)
        mask = read_image(mask_path).to(dtype=torch.int32)
        
        # instances are encoded as different colors
        obj_ids = torch.unique(mask).to(dtype=torch.int32)
        valid_ids = obj_ids[(obj_ids > 0) & (obj_ids < 10000)].to(dtype=torch.int32)

        if self.bHandleIgnore:
            ignore_ids = obj_ids[obj_ids >= 10000].to(dtype=torch.int32)
            all_ids = torch.cat([valid_ids, ignore_ids])
        else:
            all_ids = valid_ids
                
        # Create masks and boxes only for these valid IDs
        masks_np = [(mask == obj_id).to(torch.uint8).numpy().squeeze() for obj_id in all_ids]
        boxes = masks_to_boxes(torch.as_tensor(np.array(masks_np)))
        labels = (all_ids // 1000).to(torch.int64)
        if self.bHandleIgnore:
            labels[labels >= 3] = 1

        iscrowd = torch.zeros((len(all_ids),), dtype=torch.int64)
        if self.bHandleIgnore:
            iscrowd[len(valid_ids):] = 1

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        valid_box_mask = (widths > 1.0) & (heights > 1.0)
        
        boxes = boxes[valid_box_mask]
        labels = labels[valid_box_mask]
        masks_np_filtered = []
        for i, b in enumerate(valid_box_mask):
            if (b): masks_np_filtered.append(masks_np[i])
        iscrowd = iscrowd[valid_box_mask]

        # Apply Albumentations
        if self.bTransforms:
            img_np = img.permute(1, 2, 0).numpy().astype(np.uint8)

            augmented = self.transforms(
                image=img_np,
                masks=masks_np_filtered,
                bboxes=boxes.tolist(),
                class_labels=labels.tolist(),
                crowd_flags=iscrowd.tolist(),
            )
            
            img_final = augmented['image']
            if len(augmented['masks']) > 0:
                masks = torch.as_tensor(np.stack(augmented['masks']), dtype=torch.uint8)
                has_pixels = torch.any(masks.flatten(1), dim=1)
                if torch.any(has_pixels):
                    masks = masks[has_pixels]
                    labels = labels[has_pixels]
                    iscrowd = iscrowd[has_pixels]
                    boxes = masks_to_boxes(masks)
                else:
                    # All masks were empty/zeros
                    masks = torch.zeros((0, img_final.shape[1], img_final.shape[2]), dtype=torch.uint8)
                    boxes = torch.zeros((0, 4), dtype=torch.float32)
                    labels = torch.tensor([], dtype=torch.int64)
                    iscrowd = torch.tensor([], dtype=torch.int64)
            else:
                masks = torch.zeros((0, img_final.shape[1], img_final.shape[2]), dtype=torch.uint8)
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.tensor([], dtype=torch.int64)
                iscrowd = torch.tensor([], dtype=torch.int64)            
        else:
            img_final = img.to(dtype=torch.float32)
            masks = torch.as_tensor(np.array(masks_np), dtype=torch.uint8)

        if len(boxes) > 0:
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            valid_box_mask = (widths > 1.0) & (heights > 1.0)

            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            boxes = boxes[valid_box_mask]
            labels = labels[valid_box_mask]
            area = area[valid_box_mask]
            masks = masks[valid_box_mask]
            iscrowd = iscrowd[valid_box_mask]
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.tensor([], dtype=torch.int64)
            masks = torch.zeros((0, img_final.shape[1], img_final.shape[2]), dtype=torch.uint8)
            area = torch.tensor([], dtype=torch.float32)
            iscrowd = torch.tensor([], dtype=torch.int64)
        
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.functional.get_size(img_final)),
            "masks": tv_tensors.Mask(masks),
            "labels": labels.to(dtype=torch.int64),
            "image_id": idx,
            "area": area,
            "iscrowd": iscrowd
        }

        return img_final / 255.0, target

    def __len__(self):
        return len(self.img_paths)