# MCV-C5 – Group 7

This repository contains the code for the MCV-C5 group project.

> **Note:** Heavy files such as trained `.pt` models and full datasets are **not included** in this repository (they are stored in the cluster). Some links or logs are provided.

---

## Repository Structure

### 📂 `domainshift/`
Implementation of **Task F – Domain Shift** experiments.

Main files:
- `task_f.py` – Main training script  
- `deart_dataset.py` – Custom dataset  
- `engine.py`, `utils.py` – Training utilities  
- `coco_eval.py`, `coco_utils.py` – Evaluation utilities  
- `albumentations_aug.py` – Data augmentation  
- `validate_models.py` – Model validation script  
- `*.sh` – Cluster job scripts  
- `model/link_to_model.txt` – Link to externally stored trained model  

Subfolder:
- `torchvision_trial/` – Alternative torchvision-based implementation  

Validation logs (`.out`, `.err`) are included for reference.

---

### 📂 `huggingface/`
Experiments using Hugging Face models.

- `task_c/c.py` – Task C  
- `task_d/d.py` – Task D  
  - `plots/` – Evaluation plots (PR curves, mAP, etc.)  
- `task_e/task_e.py` – Task E  
- `job.sh` – Cluster job script  
- `model/` – Training output logs  

Each task folder includes its corresponding dataset and utility files.

---

### 📂 `torchvision/`
Torchvision-based implementations.

- `task_c/task_c.py`  
- `task_d/`  
- `task_e/`  

Each task contains:
- Dataset definition  
- Training script  
- COCO evaluation utilities  

---

### 📂 `ultralytics/`
YOLO (Ultralytics) experiments.

- `task_c/`, `task_d/`, `task_e/` – Task implementations  
- `kitti_to_yolo.py` – Dataset conversion script (dataset in YOLO format can be found in the cluster) 
- `kitti.yaml` – YOLO dataset configuration  
- `validate/` – Validation scripts and outputs for all tasks
- `plots/` – Experiment plots (LR, augmentation, batch size, image size, etc.)  
- `groundtruth_check/` – Ground-truth verification utilities  
- `jobs/` – Cluster job scripts  

> `yolo26n.pt` is included as a base model only. Final trained weights are stored in the cluster.

---

### 📂 `job_templates/`
Generic SLURM job templates:
- `mtgpuhigh.sh`  
- `mtgpulow.sh`  

---

## Root Files

- `requirements.txt` – Python dependencies  
- `README.md` – This file  

---