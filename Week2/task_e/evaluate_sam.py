import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

gt_file = "/ghome/group07/MCV-C5-Group7/Week2/task_c/ground_truth/kitti_mots_gt_coco_filtered.json"
pred_file = "/ghome/group07/MCV-C5-Group7/Week2/task_e/sam_predictions_pretrained.json"

# load and get valid image ids from GT
coco_gt = COCO(gt_file)
gt_img_ids = set(coco_gt.getImgIds())

# load and filter predictions
with open(pred_file) as f:
    preds = json.load(f)

preds_filtered = [p for p in preds if p["image_id"] in gt_img_ids]

print(f">>> IMPORTANT! -> Removed {len(preds) - len(preds_filtered)} predictions not in GT")

coco_dt = coco_gt.loadRes(preds_filtered)

# run evaluation
coco_eval = COCOeval(coco_gt, coco_dt, "segm")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()


# =============================
# EXTRA METRICS
# =============================

def compute_metrics(coco_eval, category_id=None):

    precision = coco_eval.eval["precision"]   # [TxRxKxAxM]
    recall = coco_eval.eval["recall"]         # [TxKxAxM]

    if category_id is not None:
        precision = precision[:, :, category_id, :, :]
        recall = recall[:, category_id, :, :]

    precision = precision[precision > -1]
    recall = recall[recall > -1]

    P = np.mean(precision)
    R = np.mean(recall)

    F1 = 2 * (P * R) / (P + R + 1e-8)

    return P, R, F1


# category ids
cats = coco_gt.loadCats(coco_gt.getCatIds())
cat_name_to_id = {c["name"]: i for i, c in enumerate(cats)}

car_id = cat_name_to_id.get("car", 0)
person_id = cat_name_to_id.get("person", 1)

# compute metrics
P_all, R_all, F1_all = compute_metrics(coco_eval)

P_car, R_car, F1_car = compute_metrics(coco_eval, car_id)
P_person, R_person, F1_person = compute_metrics(coco_eval, person_id)

# mAP
mAP50 = coco_eval.stats[1]
mAP5095 = coco_eval.stats[0]


# =============================
# PRINT TABLE
# =============================

print("\n")
print("Metric            Car        Person      All")
print("------------------------------------------------")

print(f"Precision       {P_car:.3f}      {P_person:.3f}      {P_all:.3f}")
print(f"Recall          {R_car:.3f}      {R_person:.3f}      {R_all:.3f}")
print(f"F1 score        {F1_car:.3f}      {F1_person:.3f}      {F1_all:.3f}")
print(f"mAP@50          {mAP50:.3f}      {mAP50:.3f}      {mAP50:.3f}")
print(f"mAP@50-95       {mAP5095:.3f}      {mAP5095:.3f}      {mAP5095:.3f}")