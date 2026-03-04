import tqdm

import torch

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import torchvision.transforms.v2 as F

import utils

from task_e_dataset import KittiMotsDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from engine import train_one_epoch, evaluate

"""---------------------------------------------------------------"""
def get_transform(enable=True):
    transforms = []
    
    if enable:        
        return A.Compose([
            A.HueSaturationValue(
                hue_shift_limit=3, 
                sat_shift_limit=178, 
                val_shift_limit=102, 
                p=1.0),

            A.Affine(
                translate_percent=0.1,
                shear=(-10, 10),
                p=1.0
            ),

            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.05),

            A.RandomBrightnessContrast(p=0.5),
            A.ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels', 'crowd_flags'], min_area=1.0,min_visibility=0.1,check_each_transform=True))
    else:
        transforms.append(F.ToDtype(torch.float, scale=False))
        transforms.append(F.ToPureTensor())
        return F.Compose(transforms)

"""---------------------------------------------------------------"""
def get_model_instance(num_classes):
    # load an instance detection model pre-trained on COCO
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights, box_score_thresh=0.01).to(device)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

"""-------------------------------------------------------------------"""
"""-----------------------------MAIN----------------------------------"""
"""-------------------------------------------------------------------"""
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3
    db_path = "/ghome/mcv/datasets/C5/KITTI-MOTS"
    
    # 1. Define the learning rates to test
    learning_rates = [0.0001, 0.001, 0.01]

    # 2. Outer loop for Learning Rate experiments
    for aug in [False, True]:
        for bHandleIgnore in [False, True]:      
            # Load datasets once outside the loop to save time
            train_set = KittiMotsDataset(db_path, get_transform(enable=aug), aug, bHandleIgnore, is_validation=False)
            val_set = KittiMotsDataset(db_path, get_transform(enable=False), False, bHandleIgnore, is_validation=True)

            train_data_loader = torch.utils.data.DataLoader(
                train_set, batch_size=20, shuffle=True, collate_fn=utils.collate_fn, num_workers=3
            )
            val_data_loader = torch.utils.data.DataLoader(
                val_set, batch_size=1, shuffle=False, collate_fn=utils.collate_fn, num_workers=3
            )
        
            for lr in learning_rates:
                print(f"\n--- STARTING EXPERIMENT WITH AUG: {aug} LR: {lr} HandleIgnore: {bHandleIgnore} ---")
                
                # IMPORTANT: Re-initialize model so weights reset for each LR
                model = get_model_instance(num_classes)
                model.to(device)
                
                # 3. Setup AdamW with the current learning rate
                params = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(
                    params,
                    lr=lr,
                    weight_decay=0.0005
                )
                
                num_epochs = 5
                for epoch in range(num_epochs):
                    # Train
                    train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=100)
                    
                    # Test
                    evaluate(model, val_data_loader, device=device)
                print(f"Finished experiment for LR {lr}\n")
        
            print("All experiments completed!")
