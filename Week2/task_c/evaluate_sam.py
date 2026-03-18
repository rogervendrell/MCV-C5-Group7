import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

gt_file = "/ghome/group07/MCV-C5-Group7/Week2/task_c/ground_truth/kitti_mots_gt_coco_filtered.json"
pred_file = "/ghome/group07/MCV-C5-Group7/Week2/task_b/predictions/groundedsam_predictions_01.json"

# load and get valid image ids from GT
coco_gt = COCO(gt_file)
gt_img_ids = set(coco_gt.getImgIds())

# load and filter predictions
with open(pred_file) as f:
    preds = json.load(f)
preds_filtered = [p for p in preds if p["image_id"] in gt_img_ids]

print(f">>> IMPORTANT! -> Removed {len(preds) - len(preds_filtered)} predictions not in GT")

# Evaluate
coco_dt = coco_gt.loadRes(preds_filtered)
coco_eval = COCOeval(coco_gt, coco_dt, "segm")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()