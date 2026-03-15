import json
from collections import defaultdict, Counter
from pathlib import Path

ANN_FILE = "/ghome/group07/MCV-C5-Group7/Week2/task_f/dataset/annotations.json"

def load_annotations():
    with open(ANN_FILE, "r") as f:
        data = json.load(f)

    annotations = data["annotations"]
    images = data["images"]

    # count annotations per image_id
    ann_count_per_image = defaultdict(int)
    for ann in annotations:
        ann_count_per_image[ann["image_id"]] += 1

    # mapping file_name -> image_id
    file_to_image_id = {img["file_name"]: img["id"] for img in images}

    # ensure images with 0 annotations exist
    for img in images:
        ann_count_per_image.setdefault(img["id"], 0)

    return ann_count_per_image, file_to_image_id


def print_instance_distribution(ann_count_per_image):
    distribution = Counter(ann_count_per_image.values())

    print("Instance distribution:\n")
    for n in sorted(distribution):
        print(f"images with {n} instance(s): {distribution[n]}")


def get_instance_count(image_file, ann_count_per_image, file_to_image_id):
    """
    Returns the number of instances for a given image file.
    image_file can be a full path or just the filename.
    """
    image_name = Path(image_file).name

    if image_name not in file_to_image_id:
        raise ValueError(f"{image_name} not found in annotations")

    img_id = file_to_image_id[image_name]
    return ann_count_per_image.get(img_id, 0)


def main():
    ann_count_per_image, file_to_image_id = load_annotations()

    print_instance_distribution(ann_count_per_image)

    # example usage
    test_image = "00008900.jpg"
    try:
        count = get_instance_count(test_image, ann_count_per_image, file_to_image_id)
        print(f"\n{test_image} has {count} instance(s)")
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()