# MCV-C5 – Group 7

This repository contains the implementation for the **MCV-C5 Computer Vision project**.

The project explores **object detection and segmentation methods** using multiple frameworks and modern foundation models:

* **Ultralytics YOLO**
* **Torchvision detection models (Faster R-CNN)**
* **HuggingFace models (DETR / RT-DETR)**
* **Segment Anything Model (SAM)**

Experiments are organized **by week and task** following the course assignment structure.

> **Note**
> Large datasets and trained model weights (`.pt`, `.pth`) are **not stored in this repository**.
> They are located on the **cluster**, and only logs or reference links are included.

---

# Repository Structure

```
.
├── W01                # Week 1: Object Detection
├── W02                # Week 2: SAM and Segmentation
├── job_templates      # SLURM templates for cluster jobs
├── requirements.txt
└── README.md
```

---

# Week 1 – Object Detection

Week 1 focuses on **object detection models** and their comparison on the **KITTI-MOTS dataset**.

Models explored:

* **Faster R-CNN** (Torchvision)
* **DETR / RT-DETR** (HuggingFace)
* **YOLO (>v8)** (Ultralytics)

The goal is to analyze **performance, robustness, and inference characteristics** across architectures.

## Week 1 Tasks

**a. Environment setup**

* Install and configure the required frameworks.
* Prepare the development environment and cluster job scripts.

**b. Dataset and framework exploration**

* Explore the **KITTI-MOTS dataset**.
* Familiarize with:

  * HuggingFace detection models
  * Ultralytics YOLO framework.

**c. Inference with pre-trained models**

Run inference on KITTI-MOTS using:

* **Faster R-CNN**
* **DETR**
* **YOLO (>v8)**

**d. Evaluation of pre-trained models**

Evaluate the detectors using standard object detection metrics:

* mAP
* Precision / Recall
* Per-class performance

**e. Fine-tuning on KITTI-MOTS**

Fine-tune object detection models on the dataset.

Framework-specific modifications:

* **Faster R-CNN / DETR**

  * Data augmentation with **Albumentations**

* **YOLO**

  * Data augmentation through **YOLO configuration**

**f. Domain shift experiment**

Fine-tune one detection model on a **different dataset** to analyze domain shift.

Dataset used:

* **DeART**

**g. Model comparison**

Analyze differences between models including:

* inference speed
* model size and parameters
* detection performance
* robustness to variations

**h. RT-DETR fine-tuning**

Repeat the analysis including **fine-tuned RT-DETR**.

---

<details>
<summary><b>Week 1 Implementation Details</b></summary>

## Ultralytics (YOLO)

`W01/ultralytics/`

YOLO-based experiments.

Includes:

* `task_c/` – Pretrained inference
* `task_d/` – Model evaluation
* `task_e/` – Fine-tuning experiments

Additional utilities:

* `kitti_to_yolo.py` – Convert KITTI dataset to YOLO format
* `kitti.yaml` – Dataset configuration

Analysis tools:

* `plots/`

  * Learning rate experiments
  * Data augmentation comparisons
  * Batch size experiments
  * Image size experiments

Validation:

* `validate/` – Validation scripts and cluster outputs

Annotation verification:

* `groundtruth_check/`

---

## Torchvision (Faster R-CNN)

`W01/torchvision/`

Torchvision implementations of detection tasks.

Includes:

* `task_c/` – Inference with pretrained model
* `task_d/` – Evaluation pipeline
* `task_e/` – Fine-tuning with augmentations

Utilities include:

* COCO evaluation tools
* Dataset loaders
* training utilities

---

## HuggingFace (DETR / RT-DETR)

`W01/huggingface/`

Experiments using HuggingFace detection models.

Tasks implemented:

* `task_c/` – Pretrained inference
* `task_d/` – Evaluation
* `task_e/` – Fine-tuning

Evaluation visualizations include:

* PR curves
* mAP plots
* confidence vs F1 curves

---

## Domain Shift Experiments

`W01/domainshift/`

Experiments exploring **cross-dataset generalization**.

Main scripts:

* `task_f.py` – Fine-tuning with domain shift
* `validate_models.py` – Model evaluation
* `deart_dataset.py` – DeART dataset loader

Additional training utilities:

* `engine.py`
* `utils.py`
* `albumentations_aug.py`

</details>

---

# Week 2 – Segmentation with SAM

Week 2 focuses on **instance segmentation using the Segment Anything Model (SAM)** and prompt-based segmentation.

Experiments explore how different prompts affect segmentation quality.

## Week 2 Tasks

**a. SAM inference with prompts**

Run inference with **pre-trained SAM** on the KITTI-MOTS dataset.

Evaluate segmentation results using different prompt types:

* bounding boxes
* points
* masks

---

**b. Grounded SAM with text prompts**

Use **Grounded SAM** to perform segmentation based on **text prompts**.

Example prompts:

* `"car"`
* `"person"`

Evaluate results on KITTI-MOTS.

---

**c. SAM with detection model prompts**

Use the **bounding boxes produced by the best object detection model from Week 1** as prompts for SAM.

Pipeline:

1. Run object detection model
2. Extract bounding boxes
3. Provide boxes as prompts to SAM

---

**d. Compare segmentation pipelines**

Compare the segmentation quality of:

* **Grounded SAM (text prompts)**
* **SAM using detection bounding boxes**

Metrics and qualitative comparisons are analyzed.

---

**e. Fine-tuning SAM**

Fine-tune the **Prompt Decoder of SAM** for **instance segmentation** on KITTI-MOTS.

---

**f. Domain shift experiment**

Evaluate both:

* **Pretrained SAM**
* **Fine-tuned SAM**

on another dataset to analyze generalization.

Example dataset:

* **DeART**

---

**g. Prompt analysis**

Analyze how segmentation performance varies with:

* prompt type
* prompt quality
* prompt source

---

**h. (Optional) Semantic segmentation**

Explore performing **semantic segmentation on KITTI-MOTS** using SAM.

---

<details>
<summary><b>Week 2 Implementation Details</b></summary>

## Task A

`W02/task_a/`

Basic SAM inference experiments and initial evaluation.

---

## Task B

`W02/task_b/`

Evaluation utilities and result analysis tools.

Includes:

* plotting utilities
* result aggregation
* experiment visualization

---

## Task C – Detection + SAM Pipeline

`W02/task_c/`

Pipeline combining **object detection models with SAM segmentation**.

Main functionality:

* Generate bounding boxes with YOLO
* Use boxes as SAM prompts
* Evaluate segmentation results

Includes:

* bounding box generation
* evaluation scripts
* visualization tools

---

## Task E – SAM Fine-Tuning

`W02/task_e/`

Fine-tuning the **prompt decoder of SAM**.

Includes:

* dataset loaders
* inference scripts
* evaluation scripts
* visualization tools

---

## Task F – Domain Shift with SAM

`W02/task_f/`

Evaluation of **pretrained vs fine-tuned SAM** on a different dataset.

Includes:

* dataset download scripts
* inference pipelines
* visualization tools

Dataset used:

* **DeART**

---

## Task H

`W02/task_h/`

Additional experiments and analysis.

</details>

---

# Job Templates

`job_templates/`

Reusable **SLURM templates** used to launch experiments on the cluster.

Available templates:

* `mtgpuhigh.sh`
* `mtgpulow.sh`

---

# Requirements

Dependencies are listed in:

```
requirements.txt
```

Install them with:

```bash
pip install -r requirements.txt
```

---

# Notes

* Experiments were executed on **GPU clusters using SLURM jobs**.
* `.out` and `.err` logs are included for **reproducibility and debugging**.
* Large model checkpoints and datasets are stored **externally on the cluster**.
