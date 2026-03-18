import os
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from PIL import Image

IMAGES_DIR = Path("/ghome/mcv/datasets/C5/KITTI-MOTS/training/image_02/")
YOLO_WEIGHTS = "/ghome/group07/MCV-C5-Group7/Week2/task_c/bestyolo26.pt"
YOLO_BBOXES = Path("/ghome/group07/MCV-C5-Group7/Week2/task_c/yolo_bboxes/")

BATCH_SIZE = 16


def process_batch(yolo_model, images, conf_thr):
    return yolo_model(images, classes=[0, 2], conf=conf_thr)


if __name__ == "__main__":
    yolo = YOLO(YOLO_WEIGHTS)
    all_img_seqs = sorted(os.listdir(IMAGES_DIR))

    for conf in [0.01, 0.2, 0.5, 0.7, 0.9]:
        conf_dir = YOLO_BBOXES / f"conf{conf:.2f}"

        for seq in all_img_seqs:
            seq_img_dir = IMAGES_DIR / seq
            images = sorted(seq_img_dir.glob("*.png"))

            seq_out_dir = conf_dir / seq
            seq_out_dir.mkdir(parents=True, exist_ok=True)

            for i in tqdm(range(0, len(images), BATCH_SIZE),
                          desc=f"Processing seq={seq} conf={conf}"):
                batch_paths = images[i:i+BATCH_SIZE]

                # load batch
                batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]

                results = process_batch(yolo, batch_imgs, conf)
                for img_p, r in zip(batch_paths, results):
                    out_file = seq_out_dir / f"{img_p.stem}.txt"

                    boxes = []
                    if r.boxes is not None:
                        boxes = r.boxes.xyxy.cpu().numpy().tolist()

                    label_map = {0: "person", 2: "car"}

                    with open(out_file, "w") as f:
                        if r.boxes is not None:
                            boxes = r.boxes.xyxy.cpu().numpy().tolist()
                            classes = r.boxes.cls.cpu().numpy().astype(int).tolist()
                            scores = r.boxes.conf.cpu().numpy().tolist()

                            for cls, score, b in zip(classes, scores, boxes):
                                label = label_map.get(cls)
                                if label:
                                    f.write(f"{label} {score} {' '.join(map(str, b))}\n")