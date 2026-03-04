import time

import torch

from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms.v2 as F

import utils

from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

from task_d_dataset import KittiMotsDataset

@torch.inference_mode()
def evaluate(model, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    coco_evaluator = CocoEvaluator(coco, ["bbox"])

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    coco_evaluator.summarize_per_class()
    return coco_evaluator

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transformation = F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=None),
    ])
    
    db_path = "/ghome/mcv/datasets/C5/KITTI-MOTS"
    dataset = KittiMotsDataset(db_path, transformation)
    
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=utils.collate_fn
    )
    
    # Initialize pretrained model
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT", box_score_thresh=0.01).to(device)
    
    evaluate(model, data_loader, device=device)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    