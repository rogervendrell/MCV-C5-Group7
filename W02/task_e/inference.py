import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils

from sam_utils import load_sam, get_transform, preprocess_image
import config


GT_FILE = "/ghome/group07/MCV-C5-Group7/Week2/task_c/ground_truth/kitti_mots_gt_coco_filtered.json"
FILTERED_GT_FILE = "gt_filtered_val_sequences.json"
OUT_FILE = "sam_predictions_pretrained.json"
DATASET_DIR = "/ghome/group07/mcv/datasets/C5/KITTI-MOTS/training/image_02"

# CHECKPOINT = "/ghome/group07/MCV-C5-Group7/Week2/task_e/sam_kitti_decoder_6.pth"

VAL_SEQUENCES = {
    "0000", "0001", "0002", "0003", "0004", "0005", "0006", "0007",
    "0008", "0009", "0010", "0011", "0012", "0013", "0014", "0015",
    "0016", "0017", "0018", "0019", "0020", "0021", "0022", "0023",
    "0024", "0025", "0026", "0027"
}

def main():

    device = config.DEVICE

    coco = COCO(GT_FILE)
    img_ids = coco.getImgIds()

    # Prepare filtered GT while keeping the filtering logic in the loop style
    filtered_images = []
    filtered_annotations = []
    filtered_img_ids = set()

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        seq = img_info["file_name"].split("/")[0]

        if seq not in VAL_SEQUENCES:
            continue

        filtered_images.append(img_info)
        filtered_img_ids.add(img_id)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        filtered_annotations.extend(anns)

    filtered_gt = {
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": coco.dataset["categories"]
    }

    with open(FILTERED_GT_FILE, "w") as f:
        json.dump(filtered_gt, f)

    print("Saved filtered GT to:", FILTERED_GT_FILE)
    print("Filtered images:", len(filtered_images))
    print("Filtered annotations:", len(filtered_annotations))

    # Load SAM
    sam = load_sam(
        "vit_b",
        "sam_vit_b_01ec64.pth",
        device
    )

    # Load your finetuned decoder
    # sam.mask_decoder.load_state_dict(torch.load(CHECKPOINT, map_location=device))

    sam.eval()

    transform = get_transform()

    predictions = []

    for img_id in tqdm(img_ids, desc="Running inference"):

        img_info = coco.loadImgs(img_id)[0]
        seq = img_info["file_name"].split("/")[0]
        if seq not in VAL_SEQUENCES:
            continue

        img_path = f"{DATASET_DIR}/{img_info['file_name']}"
        image = np.array(Image.open(img_path).convert("RGB"))

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        if len(anns) == 0:
            continue

        for ann in anns:

            bbox = ann["bbox"]  # COCO format xywh

            x, y, w, h = bbox
            box = np.array([x, y, x + w, y + h])

            input_image, original_size = preprocess_image(
                image,
                sam,
                transform,
                device
            )

            box_transformed = transform.apply_boxes(
                box[None, :],
                image.shape[:2]
            )

            box_torch = torch.tensor(box_transformed).float().to(device)

            with torch.no_grad():

                image_embedding = sam.image_encoder(input_image)

                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points=None,
                    boxes=box_torch,
                    masks=None
                )

                low_res_masks, _ = sam.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )

                masks = sam.postprocess_masks(
                    low_res_masks,
                    input_image.shape[-2:],
                    original_size
                )

            mask = masks[0, 0].cpu().numpy()
            binary_mask = (mask > 0).astype(np.uint8)

            if binary_mask.sum() == 0:
                continue

            rle = mask_utils.encode(np.asfortranarray(binary_mask))
            rle["counts"] = rle["counts"].decode("utf-8")

            area = int(mask_utils.area(rle))
            bbox = mask_utils.toBbox(rle).tolist()

            predictions.append({
                "image_id": img_id,
                "category_id": ann["category_id"],
                "segmentation": rle,
                "score": 1.0,
                "area": area,
                "bbox": bbox
            })

    with open(OUT_FILE, "w") as f:
        json.dump(predictions, f)

    print("Saved predictions to:", OUT_FILE)
    print("Total predictions:", len(predictions))


if __name__ == "__main__":
    main()