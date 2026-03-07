# albumentations_aug.py

import numpy as np
import torch
import albumentations as A
from torchvision import tv_tensors

def _clip_and_filter(bbs, labs, mks, H2, W2):
    if len(bbs) == 0:
        return [], [], []

    b = np.array(bbs, dtype=np.float32)

    # clip
    b[:, 0] = np.clip(b[:, 0], 0, W2 - 1)
    b[:, 2] = np.clip(b[:, 2], 0, W2 - 1)
    b[:, 1] = np.clip(b[:, 1], 0, H2 - 1)
    b[:, 3] = np.clip(b[:, 3], 0, H2 - 1)

    # enforce x1<x2, y1<y2
    x1 = np.minimum(b[:, 0], b[:, 2])
    x2 = np.maximum(b[:, 0], b[:, 2])
    y1 = np.minimum(b[:, 1], b[:, 3])
    y2 = np.maximum(b[:, 1], b[:, 3])
    b = np.stack([x1, y1, x2, y2], axis=1)

    w = b[:, 2] - b[:, 0]
    h = b[:, 3] - b[:, 1]
    keep = (w > 1.0) & (h > 1.0)

    b = b[keep]
    labs = [l for l, k in zip(labs, keep.tolist()) if k]
    mks  = [m for m, k in zip(mks,  keep.tolist()) if k]

    return b.tolist(), labs, mks

def _img_to_numpy_hwc_uint8(img: torch.Tensor) -> np.ndarray:
    """
    Espera img com tv_tensors.Image / torch.Tensor en (C,H,W).
    Retorna np.ndarray (H,W,C) uint8 per Albumentations.
    """
    if torch.is_tensor(img):
        x = img.detach().cpu()
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.permute(1, 2, 0)  # HWC
        x = x.numpy()
    else:
        x = np.array(img)

    # El teu dataset llegeix amb read_image -> int32 0..255 normalment
    if x.dtype != np.uint8:
        # si ve en int32 amb rang 0..255, només cast
        if np.issubdtype(x.dtype, np.integer):
            x = np.clip(x, 0, 255).astype(np.uint8)
        else:
            # si ve float 0..1
            x = np.clip(x, 0.0, 1.0)
            x = (x * 255.0).astype(np.uint8)

    # assegurem 3 canals
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)

    return x


def get_transforms(enable: bool):
    """
    Retorna un callable compatible amb el dataset:
      (img: tv_tensors.Image, target: dict) -> (img, target)

    Si enable=True: aplica aug a image + bboxes + masks.
    Si enable=False: només converteix i re-empaqueta per garantir consistència.
    """

    if enable:
        aug = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.10,
                    rotate_limit=10,
                    border_mode=0,
                    p=0.3,
                ),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",           # XYXY
                label_fields=["category_ids"], # labels associats a bboxes
                min_area=1.0,
                min_visibility=0.0,
            ),
        )
    else:
        # Important: igualment fem servir A.Compose amb bbox_params
        # per poder cridar aug(...) amb bboxes/masks encara que no faci res
        aug = A.Compose(
            [],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["category_ids"],
                min_area=1.0,
                min_visibility=0.0,
            ),
        )

    def _apply(img, target):
        # --- extreure info del target
        boxes = target["boxes"]
        masks = target["masks"]
        labels = target["labels"]

        # img -> numpy HWC uint8
        img_np = _img_to_numpy_hwc_uint8(img)

        # boxes -> llista de llistes (Albumentations)
        if torch.is_tensor(boxes):
            boxes_t = boxes.detach().cpu().to(torch.float32)
        else:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)

        bboxes = boxes_t.tolist() if boxes_t.numel() > 0 else []
        category_ids = labels.detach().cpu().tolist() if torch.is_tensor(labels) else list(labels)

        # masks -> llista de 2D arrays
        if torch.is_tensor(masks):
            masks_t = masks.detach().cpu()
            masks_list = [masks_t[i].numpy().astype(np.uint8) for i in range(masks_t.shape[0])]
        else:
            # per si ve com llista ja
            masks_list = [np.asarray(m).astype(np.uint8) for m in masks]

        # --- Albumentations (named args!)
        out = aug(
            image=img_np,
            bboxes=bboxes,
            masks=masks_list,
            category_ids=category_ids,
        )

        img_np2 = out["image"]
        bboxes2 = out["bboxes"]
        masks2 = out["masks"]
        labels2 = out["category_ids"]

        # després de: img_np2 = out["image"]; bboxes2 = out["bboxes"]; ...

        H2, W2 = img_np2.shape[0], img_np2.shape[1]

        bboxes2, labels2, masks2 = _clip_and_filter(bboxes2, labels2, masks2, H2, W2)

        # --- reconstruir tensores
        img_t = torch.from_numpy(img_np2).permute(2, 0, 1).contiguous()  # C,H,W
        img_t = img_t.to(torch.uint8)

        H, W = img_t.shape[-2], img_t.shape[-1]

        if len(bboxes2) > 0:
            boxes2_t = torch.tensor(bboxes2, dtype=torch.float32)
            labels2_t = torch.tensor(labels2, dtype=torch.int64)

            # masks stack
            masks2_t = torch.stack(
                [torch.from_numpy(m.astype(np.uint8)) for m in masks2],
                dim=0
            ).to(torch.uint8)

            # Filtrar caixes degenerades (area <= 0)
            area = (boxes2_t[:, 3] - boxes2_t[:, 1]) * (boxes2_t[:, 2] - boxes2_t[:, 0])
            keep = area > 0

            boxes2_t = boxes2_t[keep]
            labels2_t = labels2_t[keep]
            masks2_t = masks2_t[keep]
            area = area[keep]

        else:
            boxes2_t = torch.zeros((0, 4), dtype=torch.float32)
            labels2_t = torch.zeros((0,), dtype=torch.int64)
            masks2_t = torch.zeros((0, H, W), dtype=torch.uint8)
            area = torch.zeros((0,), dtype=torch.float32)

        # Re-empaquetar a tv_tensors
        img_tv = tv_tensors.Image(img_t)

        new_target = dict(target)  # copia shallow
        new_target["boxes"] = tv_tensors.BoundingBoxes(
            boxes2_t, format="XYXY", canvas_size=(H, W)
        )
        new_target["masks"] = tv_tensors.Mask(masks2_t)
        new_target["labels"] = labels2_t
        new_target["area"] = area
        new_target["iscrowd"] = torch.zeros((labels2_t.shape[0],), dtype=torch.int64)
        # image_id el mantenim com estava
        new_target["image_id"] = target["image_id"]

        return img_tv, new_target

    return _apply