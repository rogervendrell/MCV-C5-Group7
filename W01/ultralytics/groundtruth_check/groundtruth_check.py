import cv2
import os
import random

# Paths
images_path = "/ghome/group07/MCV-C5-Group7/ultralytics/dataset/images/val"
labels_path = "/ghome/group07/MCV-C5-Group7/ultralytics/dataset/labels/val"
output_path = "/ghome/group07/MCV-C5-Group7/ultralytics/groundtruth_check"
os.makedirs(output_path, exist_ok=True)

classes = {
    0: "person",
    2: "car",
}

def yolo_to_bbox(yolo_bbox, img_width, img_height):
    x_center, y_center, w, h = yolo_bbox
    x_center *= img_width
    y_center *= img_height
    w *= img_width
    h *= img_height
    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = int(x_center + w / 2)
    y2 = int(y_center + h / 2)
    return x1, y1, x2, y2


all_images = [f for f in os.listdir(images_path) if f.endswith((".jpg", ".png"))]
sample_images = random.sample(all_images, min(15, len(all_images)))

for img_file in sample_images:
    img_path = os.path.join(images_path, img_file)
    label_file = os.path.splitext(img_file)[0] + ".txt"
    label_path = os.path.join(labels_path, label_file)

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Draw bbox
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                bbox = list(map(float, parts[1:]))
                x1, y1, x2, y2 = yolo_to_bbox(bbox, w, h)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, classes[class_id], (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save
    cv2.imwrite(os.path.join(output_path, img_file), img)

print(f"Saved {len(sample_images)} annotated images to '{output_path}'")