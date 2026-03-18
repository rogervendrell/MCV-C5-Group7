import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.v2 as F
from torchvision.transforms.v2.functional import to_pil_image


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transformation = F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
    ])
    
    # Load data for inference
    data = ImageFolder(
        "/ghome/mcv/datasets/C5/KITTI-MOTS/testing/image_02",
        transform=transformation
    )
    
    data_loader = DataLoader(data, batch_size=1, shuffle=False)

    # Initialize pretrained model
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights, box_score_thresh=0.5).to(device)
    model.eval()

    # Initialize the inference transforms
    preprocess = weights.transforms()        

    for i, (img_tensor, _) in enumerate(data_loader):
        
        # Apply inference preprocessing transforms
        input_img = preprocess(img_tensor.squeeze(0))
        
        # Inference
        prediction = model(input_img.unsqueeze(0).to(device))[0]
        
        # Get labels from COCO weights
        labels = [weights.meta["categories"][i] for i in prediction["labels"]]
        
        # Draw the boxes on the image
        img_int = (img_tensor.squeeze(0) * 255).type(torch.uint8)
        result_img = draw_bounding_boxes(
            img_int, 
            boxes=prediction["boxes"], 
            labels=labels, 
            colors="red", 
            width=4,
            font_size=30
        )
        
        # Save the result
        #im = to_pil_image(result_img.detach())
        #im.show()
        
        print(f"Processed image {i}")
        break
    