import json
from collections import Counter

def count_instances_per_category(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    cat_id_to_name = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    counter = Counter(ann["category_id"] for ann in data.get("annotations", []))
    counts_by_name = {
        cat_id_to_name.get(cat_id, f"id={cat_id}"): count
        for cat_id, count in counter.items()
    }
    return counts_by_name


json_file = "/export/home/group07/MCV-C5-Group7/Week2/task_c/ground_truth/kitti_mots_gt_coco.json"
counts = count_instances_per_category(json_file)

for cat, count in counts.items():
    print(f"{cat}: {count}")