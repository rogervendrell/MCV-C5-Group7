from ultralytics import YOLO
from metrics_summary import make_summary

model = YOLO("/ghome/group07/MCV-C5-Group7/ultralytics/yolo26n.pt")
metrics = model.val(
    data="/ghome/group07/MCV-C5-Group7/ultralytics/kitti.yaml",
    classes=[0,2],
    imgsz=640,
    batch=16,
    conf=0.25,
    workers=4,
    visualize=True,
)

summary = make_summary(metrics)
print(summary)