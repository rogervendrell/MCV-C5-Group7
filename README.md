# MCV-C5 – Group 7

This repository contains the implementation for the **MCV-C5 Computer Vision project**.

The project explores **object detection and segmentation approaches** across multiple frameworks and foundation models:

* **Ultralytics YOLO**
* **Torchvision detection models (Faster R-CNN)**
* **HuggingFace models (DETR / RT-DETR)**
* **Segment Anything Model (SAM)**

Experiments are organized **by week and task**, following the structure of the course assignment.

> **Note**
> Large datasets and trained models (`.pt`, `.pth`) are **not included in this repository**.
> They are stored on the **cluster**, and only logs or reference links are provided.

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

<details>
<summary><b>Click to expand Week 1 details</b></summary>

Week 1 focuses on **object detection models** and their evaluation on the **KITTI-MOTS dataset**.

Models explored:

* **Faster R-CNN** (Torchvision)
* **DETR / RT-DETR** (HuggingFace)
* **YOLO (>v8)** (Ultralytics)

The goal is to analyze **performance, robustness, and inference characteristics** across different architectures.

---

## Week 1 Tasks

**a. Environment Setup**

* Install and configure the development frameworks.
* Prepare the execution environment and cluster scripts.

---

**b. Dataset and Framework Exploration**

* Explore the **KITTI-MOTS dataset**.
* Familiarize with the **HuggingFace** and **Ultralytics** frameworks.

---

**c. Inference with Pretrained Models**

Run inference on KITTI-MOTS using:

* **Faster R-CNN**
* **DETR**
* **YOLO (>v8)**

---

**d. Evaluation of Pretrained Models**

Evaluate the detectors using standard metrics:

* mAP
* Precision / Recall
* Per-class detection performance

---

**e. Fine-tuning on KITTI-MOTS**

Fine-tune object detection models on the dataset.

Framework-specific details:

**Faster R-CNN and DETR**

* Data augmentations implemented with **Albumentations**

**YOLO**

* Data augmentations configured within the **YOLO training configuration**

---

**f. Domain Shift Experiment**

Fine-tune either **Faster R-CNN or DETR** on a different dataset to analyze domain shift.

Dataset used:

* **DeART dataset**

---

**g. Model Comparison**

Analyze the differences between object detection models including:

* number of parameters
* inference time
* model robustness
* detection performance

---

**h. RT-DETR Fine-tuning**

Repeat the previous analysis including **fine-tuned RT-DETR**.

---

# Week 1 Implementation Structure

### Ultralytics (YOLO)

`W01/ultralytics/`

YOLO-based experiments.

Contains implementations for:

* `task_c/` – Pretrained inference
* `task_d/` – Model evaluation
* `task_e/` – Fine-tuning experiments

Additional utilities:

* `kitti_to_yolo.py` – Convert KITTI dataset to YOLO format
* `kitti.yaml` – Dataset configuration

Analysis tools:

* `plots/`

  * learning rate experiments
  * data augmentation experiments
  * batch size comparisons
  * image size experiments

Validation scripts and outputs:

* `validate/`

Ground-truth verification utilities:

* `groundtruth_check/`

Cluster job scripts:

* `jobs/`

---

### Torchvision (Faster R-CNN)

`W01/torchvision/`

Torchvision implementations of the detection tasks.

Includes:

* `task_c/` – Pretrained inference
* `task_d/` – Model evaluation
* `task_e/` – Fine-tuning

Utilities include:

* dataset loaders
* training utilities
* COCO evaluation scripts

---

### HuggingFace (DETR / RT-DETR)

`W01/huggingface/`

Experiments using HuggingFace detection models.

Tasks implemented:

* `task_c/` – inference
* `task_d/` – evaluation
* `task_e/` – fine-tuning

Evaluation visualizations include:

* PR curves
* mAP plots
* F1 vs confidence curves

Cluster training is executed using:

```
job.sh
```

---

### Domain Shift Experiments

`W01/domainshift/`

Experiments analyzing **cross-dataset generalization**.

Main scripts:

* `task_f.py` – training with domain shift
* `validate_models.py` – model validation
* `deart_dataset.py` – dataset loader

Training utilities:

* `engine.py`
* `utils.py`
* `albumentations_aug.py`

Evaluation utilities:

* `coco_eval.py`
* `coco_utils.py`

</details>

---

# Week 2 – Segmentation with SAM

<details>
<summary><b>Click to expand Week 2 details</b></summary>

Week 2 focuses on **instance segmentation using the Segment Anything Model (SAM)** and prompt-based segmentation techniques.

The experiments investigate **how different prompts affect segmentation performance**.

---

## Week 2 Tasks

**a. SAM Inference with Prompts**

Run inference using **pre-trained SAM** on the **KITTI-MOTS dataset**.

Evaluate segmentation performance using different prompt types:

* bounding boxes
* points
* masks

---

**b. Grounded SAM with Text Prompts**

Use **Grounded SAM** to perform segmentation using **text prompts**.

Examples:

* `"car"`
* `"person"`

Evaluate segmentation performance on KITTI-MOTS.

---

**c. Detection + SAM Pipeline**

Use bounding boxes generated by the **best object detection model from Week 1** as prompts for SAM.

Pipeline:

1. run object detection
2. extract bounding boxes
3. provide bounding boxes as SAM prompts

---

**d. Compare Segmentation Pipelines**

Compare results between:

* **Grounded SAM (text prompts)**
* **SAM with bounding boxes from the detection model**

---

**e. Fine-tuning SAM**

Fine-tune the **Prompt Decoder of SAM** for **instance segmentation** on KITTI-MOTS.

---

**f. Domain Shift Evaluation**

Evaluate both:

* **pretrained SAM**
* **fine-tuned SAM**

on another dataset to analyze generalization.

Example dataset:

* **DeART**

---

**g. Prompt Analysis**

Analyze how segmentation quality varies with:

* prompt type
* prompt source
* prompt accuracy

---

**h. (Optional) Semantic Segmentation**

Explore performing **semantic segmentation on KITTI-MOTS** using SAM.

---

# Week 2 Implementation Structure

### Task A

`W02/task_a/`

Initial experiments running SAM inference and evaluation.

Main script:

```
task_a.py
```

---

### Task B

`W02/task_b/`

Evaluation and visualization utilities.

Includes:

* result processing
* plotting tools
* experiment analysis scripts

---

### Task C – Detection + SAM Pipeline

`W02/task_c/`

Pipeline combining **object detection outputs with SAM segmentation**.

Main components:

* bounding box generation
* segmentation evaluation
* visualization utilities

---

### Task E – SAM Fine-tuning

`W02/task_e/`

Fine-tuning the **SAM prompt decoder**.

Includes:

* dataset loaders
* inference scripts
* evaluation tools
* visualization utilities

---

### Task F – Domain Shift with SAM

`W02/task_f/`

Evaluation of **SAM generalization** to another dataset.

Includes:

* dataset download scripts
* inference pipelines
* visualization tools

Dataset used:

* **DeART**

---

### Task H

`W02/task_h/`

Additional experiments and analysis scripts.

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
* `.out` and `.err` logs are included for **debugging and experiment tracking**.
* Large model checkpoints and datasets are stored **externally on the cluster**.
