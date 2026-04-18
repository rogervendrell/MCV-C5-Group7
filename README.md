# MCV-C5 – Group 7

This repository contains the implementation for the **MCV-C5 Computer Vision project**.

The project covers **object detection, segmentation, image captioning, and synthetic data generation** across multiple frameworks and foundation models:

* **Ultralytics YOLO**
* **Torchvision detection models (Faster R-CNN)**
* **HuggingFace models (DETR / RT-DETR)**
* **Segment Anything Model (SAM)**
* **Encoder-decoder captioning models (ResNet + GRU / LSTM / Transformer)**
* **ViT + LoRA-tuned Qwen3 captioning model**
* **Stable Diffusion models (SDXL, SDXL-Turbo, SD3.5-Medium)**

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
├── Week3              # Week 3: Image Captioning
├── Week4              # Week 4: ViT + Qwen3 Captioning
├── Week5              # Week 5: Diffusion Models & Synthetic Augmentation
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

# Week 3 – Image Captioning

<details>
<summary><b>Click to expand Week 3 details</b></summary>

Week 3 focuses on **image captioning** using the **VizWiz dataset** — a collection of images taken by blind users paired with crowd-sourced captions.

The experiments progressively increase model complexity, moving from a character-level GRU baseline to word-level and subword-level tokenization, stronger encoders, LSTM decoders, and finally a Transformer decoder with attention.

---

## Week 3 Tasks

**a. Baseline: ResNet-18 + GRU (character-level)**

* Train a character-level encoder-decoder model.
* Encoder: **ResNet-18** (pretrained).
* Decoder: multi-layer **GRU**.
* Evaluate using **BLEU** and **CIDEr** metrics.

---

**b. Tokenization Variants**

Explore different tokenization strategies for the decoder:

* **Character-level** – one token per character.
* **Word-level** – standard whitespace tokenization.
* **Subword-level** – **SentencePiece** BPE tokenization.

Analyze the effect on vocabulary size, training stability, and caption quality.

---

**c. Stronger Encoder**

Replace ResNet-18 with **ResNet-34** and evaluate the impact on captioning performance.

Optionally unfreeze the last encoder stage during fine-tuning.

---

**d. LSTM Decoder**

Swap the GRU decoder for an **LSTM decoder** with dropout regularization.

Experiment with:

* hidden size
* number of layers
* encoder backbone (ResNet-18 vs ResNet-34 vs ResNet-50)

---

**e. Transformer Decoder with Attention**

Replace the recurrent decoder with a **Transformer decoder** (multi-head self-attention + cross-attention).

Key components:

* Sinusoidal **positional encoding**
* Teacher forcing during training
* Temperature-controlled sampling at inference

---

**f. Qualitative Analysis**

Generate and visualize captions for sample images.

Compare outputs across models and tokenization strategies.

---

## Week 3 Implementation Structure

### Baseline (character-level GRU)

`Week3/1-baseline/`

Character-level encoder-decoder baseline.

Main components:

* `model.py` – ResNet-18 encoder + GRU decoder
* `train.py` – training loop with teacher forcing
* `dataset.py` – VizWiz dataset loader
* `vocabulary.py` – character vocabulary
* `metrics.py` – BLEU / CIDEr evaluation
* `main.py` – entry point
* `run.sh` / `evaluate.sh` – cluster scripts

---

### Baseline (word-level GRU)

`Week3/1-baseline-wordlevel/`

Word-level variant of the baseline. Same architecture with a word vocabulary and optional regularization experiments.

---

### Baseline (subword-level GRU)

`Week3/1-baseline-subwordlevel/`

Subword tokenization using **SentencePiece BPE**.

Additional files:

* `tokenizer.py` – SentencePiece wrapper
* `sp_model.model` / `sp_model.vocab` – trained BPE model

---

### Stronger Encoder (ResNet-34 + GRU)

`Week3/2-encoder/`

Replaces ResNet-18 with ResNet-34 with optional partial unfreezing of the encoder.

---

### LSTM Decoder

`Week3/3-lstm/`

ResNet encoder paired with an **LSTM decoder** with dropout and configurable hidden size.

---

### Transformer Decoder (Attention)

`Week3/4-attention/`

ResNet encoder paired with a **Transformer decoder**.

Key components:

* Positional encoding
* Multi-head attention
* Teacher forcing + temperature sampling

Cluster script: `w3_transformer_layers.sh`

---

### Plots and Analysis

`Week3/0-plots/`

Plotting and parsing utilities for experiment results.

Organized by experiment type:

* `char-vs-subword-vs-word/` – tokenization comparison
* `encoder/` – encoder ablation
* `encoder-decoder-lstm/` – LSTM decoder results
* `encoder-decoder-transformer/` – Transformer decoder results
* `hyperparameter_search/` – learning rate and batch size sweeps
* `regularization/` – dropout and weight decay experiments
* `teacher-forcing/` – teacher forcing ratio analysis
* `temperature/` – inference temperature analysis

---

### Qualitative Analysis

`Week3/0-qualitative/`

Script to generate and display sample captions for qualitative evaluation.

</details>

---

# Week 5 – Diffusion Models & Synthetic Data Augmentation

<details>
<summary><b>Click to expand Week 5 details</b></summary>

Week 5 investigates whether **synthetic text–image pairs generated with diffusion models** can improve image captioning performance on the **VizWiz dataset**. The week covers diffusion model exploration, synthetic dataset generation, and fine-tuning the ViT + Qwen3 captioning model from Week 4 with augmented data.

---

## Week 5 Tasks

**a. Stable Diffusion Model Architecture & Variants**

* Study the **Latent Diffusion Model (LDM)** framework: VAE encoder/decoder, denoising U-Net, CLIP text encoder, and classifier-free guidance (CFG).
* Compare four model variants on direct text-to-image inference:
  * **SD-Turbo** – 1-step distilled, lowest VRAM (3 GB)
  * **SDXL** – 3× larger U-Net, dual text encoder, best quality/resource ratio at 25 steps
  * **SDXL-Turbo** – distilled SDXL, 1 step, 8 GB VRAM
  * **SD-3.5-Medium** – Multimodal Diffusion Transformer, three text encoders, highest quality, 18 GB VRAM

---

**b. Inference Experiments**

Using SDXL (best quality/resource ratio), a series of controlled experiments explore diffusion model behaviour:

* **Denoising trajectory** – latent snapshots at steps 0, 5, 10 … 50 show how global structure emerges early and fine details sharpen late.
* **DDPM vs DDIM** – stochastic (DDPM) vs deterministic ODE (DDIM) samplers across 10–100 steps.
* **Eta parameter** – interpolates between DDIM (η=0) and DDPM (η=1), trading stability for diversity.
* **Scheduler zoo** – DDIM, PNDM, Euler, Euler-A, DPM-Solver++, UniPC across 10/20/50 steps. DPM-Solver++ at 20 steps chosen as best.
* **Number of denoising steps** – quality saturation analysis; ~30 DDIM steps is the practical optimum.
* **CFG guidance scale** – 1–20 sweep; sweet spot at 7.5–10.
* **Prompt engineering** – bare / detailed positive / negative / full / style / negative-overload conditions.

---

**c. VizWiz Dataset Analysis & Problem Identification**

* Inspect the VizWiz training set (23,431 images) and identify five recurring degradation patterns: **motion blur**, **dark/underexposed**, **close-up/bad framing**, **overexposed**, **low quality/noisy**.
* Score all training images with **CLIP ViT-L/14** using a quality score `sim(image, positive) − sim(image, negative)`. Result: **97.4 % of images score negative**, confirming the dataset is inherently degraded.
* Propose generating synthetic degraded images to augment training and improve model robustness.

---

**d. Synthetic Dataset Generation**

Three generation approaches were explored:

1. **Image-to-image diffusion (discarded)** – SD-Turbo/SDXL img2img with degradation prompts. At low strength the model repairs images; at high strength semantic content is lost. Approach abandoned.

2. **Text-to-image with degradation prompts** – Drop the reference image; use the original caption as a semantic anchor and a degradation descriptor as a style modifier (e.g. `"a blurry, out of focus version of {caption}"`). Negative prompts used per style. Styles assigned round-robin by CLIP quality rank (~4,686 images per style). Better alignment with target degradations, though hallucinations remain.

3. **Caption-only (clean baseline)** – Use the raw VizWiz caption as the sole prompt, no negative prompt. Generates clean, photorealistic images as a simpler augmentation baseline.

For approaches 2 and 3: **23,431 synthetic images** each, generated with **SDXL-Turbo** (8 inference steps, 512 × 512, guidance scale 0). Annotations saved in VizWiz-compatible `annotations.json` format.

---

**e. Fine-tuning with Synthetic Data**

Fine-tune the Week 4 captioning model (frozen ViT-Task1 encoder → linear projection → LoRA Qwen3-1.7B) with 50 % augmented data mixed into the original VizWiz training split:

* **Original training samples:** 22,866
* **Augmented samples added:** 11,715 (50 %)
* **Total per run:** 34,581

Two runs executed in parallel:

| Run | Augmentation | Result |
|---|---|---|
| Degradation Aug50% | Approach 2 (degradation txt2img) | Metric instability persists; recovers at final epoch; METEOR 26.1% at best epoch |
| Caption-only Aug50% | Approach 3 (clean txt2img) | Stable training from epoch 0; METEOR 27.5% at best epoch; outperforms baseline on all metrics |

Key finding: **caption-only augmentation substantially mitigates the ViT–Qwen embedding space misalignment** identified in Week 4. The broader visual distribution regularises the linear projection layer. Degradation augmentation worsens the misalignment because the ViT encoder was not trained on artificial synthetic degradations.

---

## Week 5 Implementation Structure

### Task A – Model Variants

`Week5/task_a/`

* `compare_models.py` – run and benchmark SD-Turbo, SDXL, SDXL-Turbo, SD-3.5-Medium on a fixed prompt; records load time, generation time, and peak VRAM.

---

### Task B – Inference Experiments

`Week5/task_b/`

* `explore_inference.py` – denoising trajectory, DDPM/DDIM comparison, eta sweep, scheduler zoo, step count ablation, CFG sweep, and prompt engineering experiments. All using SDXL.

---

### Task C – Dataset Analysis

`Week5/task_c/`

* `task_c.py` – CLIP quality scoring of the VizWiz training set; produces the quality score histogram.

---

### Task D – Synthetic Dataset Generation

`Week5/task_d/`

* `task_d_img2img.py` – image-to-image pipeline (approach 1, discarded).
* `task_d.py` – text-to-image degradation pipeline (approach 2).
* `generate_augmented_dataset.py` – unified generator for both degradation mode and caption-only mode; saves VizWiz-compatible `annotations.json`.
* `clip_score.py` – CLIP quality scoring utilities.
* `augmented_dataset/` – output directories for generated datasets (not tracked in git).

---

### Task E – Fine-tuning

`Week5/task_e/`

* `model.py` – `FrozenViTEncoder` + `ViTCausalLMCaptioningModel` (frozen ViT → linear projection → LoRA Qwen3-1.7B). Only the projection layer and LoRA adapters are trainable.
* `dataset.py` – VizWiz dataset loader with augmented dataset merging (`ConcatDataset`).
* `train.py` – training loop with greedy decoding during validation, metric-based checkpoint saving, resume support (`resume.pt`).
* `metrics.py` – BLEU, ROUGE-L, METEOR, sacreBLEU evaluation.
* `main.py` – entry point; `--aug-fraction` and `--aug-dataset` control augmentation.
* `plots/` – analysis plots (training/validation loss curves, BLEU-1 stability panel, metrics comparison bar chart).
* `results/` – per-run `history.csv`, `summary.json`, `predictions_val.json`, and model weights (not tracked in git).

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
