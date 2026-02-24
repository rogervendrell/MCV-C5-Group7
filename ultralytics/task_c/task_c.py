from ultralytics import YOLO

model = YOLO("/ghome/group07/MCV-C5-Group7/ultralytics/yolo26n.pt")
results = model.val(
    data="/ghome/group07/MCV-C5-Group7/ultralytics/kitti.yaml",
    imgsz=640,
    batch=16,
    conf=0.25,
    workers=4,
)