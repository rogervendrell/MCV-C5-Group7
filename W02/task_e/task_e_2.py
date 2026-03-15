import torch
from torch.utils.data import DataLoader
from torch.nn.functional import threshold, normalize

from dataset import KITTIMOTSDataset, collate_fn
from sam_utils import load_sam, get_transform, preprocess_image
import config
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import os


DEBUG_VIS = True
DEBUG_SAMPLES = 5

def visualize_sample(image, mask, box, idx):

    os.makedirs("debug_vis", exist_ok=True)

    vis = image.copy()

    # mask
    colored_mask = np.zeros_like(vis)
    colored_mask[:,:,1] = mask * 255

    vis = np.where(mask[:,:,None], vis*0.5 + colored_mask*0.5, vis)

    # bbox
    x1, y1, x2, y2 = box.astype(int)

    vis[y1:y1+2, x1:x2] = [255,0,0]
    vis[y2:y2+2, x1:x2] = [255,0,0]
    vis[y1:y2, x1:x1+2] = [255,0,0]
    vis[y1:y2, x2:x2+2] = [255,0,0]

    plt.figure(figsize=(6,4))
    plt.imshow(vis.astype(np.uint8))
    plt.axis("off")

    plt.savefig(f"debug_vis/sample_{idx}.png", bbox_inches="tight")
    plt.close()


def train():
    dataset = KITTIMOTSDataset(config.DATASET_ROOT)

    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    sam = load_sam(
        config.MODEL_TYPE,
        config.SAM_CHECKPOINT,
        config.DEVICE
    )

    transform = get_transform()

    optimizer = torch.optim.Adam(
        sam.mask_decoder.parameters(),
        lr=config.LR
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()

    sam.train()

    for epoch in range(config.EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        
        for images, gt_masks, boxes, classes in pbar:

            batch_loss = 0

            for i in range(len(images)):

                image = images[i]
                box = boxes[i]
                gt_mask = gt_masks[i]
                cls = classes[i]

                if DEBUG_VIS and epoch == 0 and i < DEBUG_SAMPLES:
                    visualize_sample(image, gt_mask, box, i)

                input_image, original_size = preprocess_image(
                    image,
                    sam,
                    transform,
                    config.DEVICE
                )

                box = transform.apply_boxes(box[None,:], image.shape[:2])
                box_torch = torch.tensor(box).float().to(config.DEVICE)

                gt_mask = torch.tensor(gt_mask).unsqueeze(0).unsqueeze(0).float().to(config.DEVICE)

                with torch.no_grad():

                    image_embedding = sam.image_encoder(input_image)

                    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                        points=None,
                        boxes=box_torch,
                        masks=None,
                    )

                low_res_masks, _ = sam.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                masks = sam.postprocess_masks(
                    low_res_masks,
                    input_image.shape[-2:],
                    original_size
                )

                gt_mask = torch.nn.functional.interpolate(
                    gt_mask,
                    size=masks.shape[-2:],
                    mode="nearest"
                )

                loss = loss_fn(masks, gt_mask)

                batch_loss += loss

            batch_loss = batch_loss / len(images)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()

            pbar.set_postfix(loss=batch_loss.item())

        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader)}")

        torch.save(
            sam.mask_decoder.state_dict(),
            f"sam_kitti_decoder_{epoch+1}.pth"
        )


if __name__ == "__main__":
    train()