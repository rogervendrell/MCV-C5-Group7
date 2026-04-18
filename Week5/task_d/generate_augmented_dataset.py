"""
Task D – Full dataset augmentation.

Iterates over every VizWiz training image (sorted best → worst by CLIP
quality score), assigns one degradation style per image via round-robin,
generates the degraded version with a txt2img diffusion model, and saves
the result in a layout that is drop-in compatible with the VizWiz dataset.

Output layout
-------------
augmented_dataset/<run_id>/
    images/
        VizWiz_train_aug_blur_00004317.jpg
        VizWiz_train_aug_dark_00012286.jpg
        ...
    annotations.json          # VizWiz-schema JSON (images + annotations)

annotations.json schema
-----------------------
Matches the original VizWiz train.json exactly, with two extra fields per
image entry: "augmentation_style" and "original_image_id", so the
augmented set can be merged or used standalone.

Features
--------
* Batched GPU generation  (--batch_size, default 8)
* Resume / checkpoint     - already-generated images are skipped
* annotations.json is saved every --checkpoint_every images and at the end
* tqdm progress bar showing images/s and ETA

Usage
-----
python generate_augmented_dataset.py                        # all defaults
python generate_augmented_dataset.py --model sdxl --batch_size 2
python generate_augmented_dataset.py --output_dir /path/to/out --seed 0
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DDIMScheduler,
)
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

VIZWIZ_TRAIN_IMG_DIR = Path("/ghome/group07/MCV-C5-Group7/Week3/dataset/train")
VIZWIZ_TRAIN_ANN     = Path("/ghome/group07/MCV-C5-Group7/Week3/dataset/annotations/train.json")
DEFAULT_SCORES_JSON  = Path("/ghome/group07/MCV-C5-Group7/Week5/task_d/results/clip_scores.json")
DEFAULT_OUTPUT_BASE  = Path("/ghome/group07/MCV-C5-Group7/Week5/task_d/augmented_dataset")

# ---------------------------------------------------------------------------
# Degradation styles  (shared with task_d.py)
# ---------------------------------------------------------------------------

_NEG_BASE = (
    "professional photography, studio lighting, sharp focus, well-composed, "
    "high quality, DSLR, 8k, HD, crisp, clean, aesthetic"
)

DEGRADATION_STYLES = {
    "blur": {
        "prompt_template": (
            "blurry photo, severe camera shake, motion blur, out of focus, "
            "{caption}, amateur snapshot, poorly taken phone photo, blurry, "
            "unfocused, lens blur, smeared"
        ),
        "negative": f"sharp, in focus, crisp, clear, {_NEG_BASE}",
        "label":    "Blur / out-of-focus",
    },
    "dark": {
        "prompt_template": (
            "very dark underexposed photo taken in a poorly lit room, "
            "{caption}, dim lighting, dark shadows, grainy, underexposed, "
            "barely visible, low light snapshot, dark, murky"
        ),
        "negative": f"bright, well-lit, properly exposed, studio lighting, {_NEG_BASE}",
        "label":    "Dark / underexposed",
    },
    "close_up": {
        "prompt_template": (
            "accidental extreme close-up photo, too close to subject, "
            "{caption}, object fills entire frame, partially cut off, "
            "clipped edges, wrong distance, unintentional macro shot"
        ),
        "negative": f"wide shot, full frame, properly framed, full view, {_NEG_BASE}",
        "label":    "Close-up / occluded",
    },
    "overexposed": {
        "prompt_template": (
            "severely overexposed photo, blown out white highlights, harsh camera flash, "
            "{caption}, washed out, white glare, overlit, "
            "blown highlights, too much light, pale washed image"
        ),
        "negative": f"properly exposed, balanced lighting, natural light, {_NEG_BASE}",
        "label":    "Overexposed / glare",
    },
    "low_quality": {
        "prompt_template": (
            "low resolution phone camera snapshot, heavy jpeg compression artifacts, "
            "{caption}, pixelated, grainy, digital noise, "
            "poor quality photo, bad camera sensor, lossy compression, blotchy"
        ),
        "negative": f"high resolution, crisp, artifact-free, {_NEG_BASE}",
        "label":    "Low quality / noisy",
    },
}

STYLE_KEYS = list(DEGRADATION_STYLES.keys())

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "sdxl": {
        "model_id":            "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline_cls":        StableDiffusionXLPipeline,
        "dtype":               torch.float16,
        "variant":             "fp16",
        "num_inference_steps": 40,
        "guidance_scale":      5.0,
        "img_size":            1024,
    },
    "sdxl-turbo": {
        "model_id":            "stabilityai/sdxl-turbo",
        "pipeline_cls":        StableDiffusionXLPipeline,
        "dtype":               torch.float16,
        "variant":             "fp16",
        "num_inference_steps": 8,
        "guidance_scale":      0.0,
        "img_size":            512,
    },
    "sd-turbo": {
        "model_id":            "stabilityai/sd-turbo",
        "pipeline_cls":        StableDiffusionPipeline,
        "dtype":               torch.float16,
        "variant":             "fp16",
        "num_inference_steps": 8,
        "guidance_scale":      0.0,
        "img_size":            512,
    },
}

DEFAULT_MODEL = "sd-turbo"

# New image IDs start here to avoid clashing with original VizWiz IDs (0–23430)
AUG_ID_OFFSET = 100_000

# Generation modes
MODE_DEGRADATION  = "degradation"   # current behaviour: wrap caption in degradation prompt
MODE_CAPTION_ONLY = "caption_only"  # new: use the raw caption as the prompt
DEFAULT_MODE      = MODE_DEGRADATION

# Style key used in filenames / annotations for caption-only mode
PLAIN_STYLE_KEY = "plain"

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_sorted_entries(scores_json: Path) -> list[dict]:
    """Return all training entries sorted best → worst by CLIP quality score."""
    with open(scores_json) as f:
        return json.load(f)["scores"]   # already sorted descending


def load_caption_index(ann_file: Path) -> dict[int, str]:
    """Return {image_id: first non-rejected caption}."""
    with open(ann_file) as f:
        data = json.load(f)
    caps: dict[int, list[str]] = defaultdict(list)
    for ann in data["annotations"]:
        if not ann.get("is_rejected", False):
            caps[ann["image_id"]].append(ann["caption"].strip())
    return {img_id: v[0] for img_id, v in caps.items() if v}


def aug_filename(style_key: str, original_file: str) -> str:
    """VizWiz_train_aug_blur_00004317.jpg"""
    stem = Path(original_file).stem          # VizWiz_train_00004317
    num  = stem.split("_")[-1]               # 00004317
    return f"VizWiz_train_aug_{style_key}_{num}.jpg"


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompts_for_batch(batch: list[dict], mode: str) -> tuple[list[str], list[str]]:
    """Return (prompts, negatives) for a batch of work items."""
    prompts   = []
    negatives = []
    for item in batch:
        if mode == MODE_DEGRADATION:
            style = DEGRADATION_STYLES[item["style_key"]]
            prompts.append(style["prompt_template"].format(caption=item["caption"]))
            negatives.append(style["negative"])
        else:
            prompts.append(item["caption"])
            negatives.append("")   # no negative prompt in caption-only mode
    return prompts, negatives


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def load_pipeline(model_key: str, device: str):
    cfg = MODEL_REGISTRY[model_key]
    print(f"Loading {model_key} ({cfg['model_id']}) ...", flush=True)
    pipe = cfg["pipeline_cls"].from_pretrained(
        cfg["model_id"],
        torch_dtype=cfg["dtype"],
        use_safetensors=True,
        variant=cfg["variant"],
    ).to(device)
    if cfg["guidance_scale"] > 0:
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_batch(
    pipe,
    model_key: str,
    prompts: list[str],
    negatives: list[str],
    seed: int,
) -> list[Image.Image]:
    """Generate one image per prompt in a single batched pipeline call."""
    cfg = MODEL_REGISTRY[model_key]
    generator = torch.Generator(device=pipe.device.type).manual_seed(seed)

    kwargs = dict(
        prompt=prompts,
        num_inference_steps=cfg["num_inference_steps"],
        guidance_scale=cfg["guidance_scale"],
        generator=generator,
        height=cfg["img_size"],
        width=cfg["img_size"],
    )
    if cfg["guidance_scale"] > 0:
        kwargs["negative_prompt"] = negatives

    return pipe(**kwargs).images   # list of PIL images, len == len(prompts)


# ---------------------------------------------------------------------------
# Annotations helper
# ---------------------------------------------------------------------------

def save_annotations(
    ann_records: list[dict],
    img_records: list[dict],
    output_path: Path,
    model_key: str,
    run_id: str,
    mode: str = MODE_DEGRADATION,
):
    out = {
        "info": {
            "description":  "VizWiz Training Set – txt2img augmentation",
            "model":        MODEL_REGISTRY[model_key]["model_id"],
            "run_id":       run_id,
            "mode":         mode,
            "styles":       STYLE_KEYS if mode == MODE_DEGRADATION else [PLAIN_STYLE_KEY],
        },
        "images":      img_records,
        "annotations": ann_records,
    }
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Task D – generate full augmented VizWiz training dataset"
    )
    parser.add_argument("--model",       default=DEFAULT_MODEL,
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--batch_size",  type=int, default=8,
                        help="Images per pipeline call (default: 8)")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--scores_json", type=str, default=str(DEFAULT_SCORES_JSON))
    parser.add_argument("--output_dir",  type=str, default=str(DEFAULT_OUTPUT_BASE),
                        help="Base output dir; run subdir is appended automatically "
                             "when --run_id is not set")
    parser.add_argument("--run_id",      type=str, default=None,
                        help="Run identifier (defaults to SLURM_JOB_ID or 'local')")
    parser.add_argument("--checkpoint_every", type=int, default=1000,
                        help="Save annotations.json every N images (default: 1000)")
    parser.add_argument("--mode",        default=DEFAULT_MODE,
                        choices=[MODE_DEGRADATION, MODE_CAPTION_ONLY],
                        help=(
                            f"'{MODE_DEGRADATION}': wrap caption in a degradation prompt "
                            f"(blur/dark/close_up/…) — default. "
                            f"'{MODE_CAPTION_ONLY}': use the raw reference caption as-is, "
                            f"generating a clean synthetic image for each training example."
                        ))
    args = parser.parse_args()

    run_id     = args.run_id or "local"
    output_dir = Path(args.output_dir) / run_id
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_path   = output_dir / "annotations.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device       : {device}")
    if torch.cuda.is_available():
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(f"Model        : {args.model}")
    print(f"Mode         : {args.mode}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Output dir   : {output_dir}")
    print(f"Run ID       : {run_id}")
    print()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading CLIP scores ...")
    entries = load_sorted_entries(Path(args.scores_json))
    print(f"  {len(entries)} images, sorted best → worst by quality score")

    print("Loading caption index ...")
    caption_index = load_caption_index(VIZWIZ_TRAIN_ANN)

    # ------------------------------------------------------------------
    # Resume: find already-generated images and skip them
    # ------------------------------------------------------------------
    existing = {p.name for p in images_dir.glob("*.jpg")}
    print(f"Already generated: {len(existing)} images — will skip")

    # Assign style keys and filenames depending on mode.
    # degradation: round-robin across STYLE_KEYS by CLIP rank.
    # caption_only: all images use the PLAIN_STYLE_KEY sentinel.
    work_items = []
    for rank, entry in enumerate(entries):
        if args.mode == MODE_DEGRADATION:
            style_key = STYLE_KEYS[rank % len(STYLE_KEYS)]
        else:
            style_key = PLAIN_STYLE_KEY
        fname = aug_filename(style_key, entry["file_name"])
        if fname in existing:
            continue
        work_items.append({
            "rank":       rank,
            "image_id":   entry["image_id"],
            "file_name":  entry["file_name"],
            "style_key":  style_key,
            "aug_fname":  fname,
            "caption":    caption_index.get(entry["image_id"], "a photograph"),
        })

    print(f"To generate    : {len(work_items)} images\n")

    # ------------------------------------------------------------------
    # Load annotations already saved (for resume)
    # ------------------------------------------------------------------
    img_records: list[dict] = []
    ann_records: list[dict] = []

    if ann_path.exists():
        with open(ann_path) as f:
            saved = json.load(f)
        img_records = saved.get("images", [])
        ann_records = saved.get("annotations", [])
        print(f"Resuming annotations: {len(img_records)} entries loaded")

    # ------------------------------------------------------------------
    # Load pipeline
    # ------------------------------------------------------------------
    pipe = load_pipeline(args.model, device)

    # ------------------------------------------------------------------
    # Batched generation
    # ------------------------------------------------------------------
    cfg         = MODEL_REGISTRY[args.model]
    ann_id_next = AUG_ID_OFFSET + len(ann_records)
    img_id_next = AUG_ID_OFFSET + len(img_records)

    pbar = tqdm(
        total=len(work_items),
        desc="Generating",
        unit="img",
        dynamic_ncols=True,
    )

    for batch_start in range(0, len(work_items), args.batch_size):
        batch = work_items[batch_start: batch_start + args.batch_size]

        prompts, negatives = build_prompts_for_batch(batch, args.mode)
        images = generate_batch(pipe, args.model, prompts, negatives, args.seed)

        for item, img in zip(batch, images):
            # Save image as JPEG (matches VizWiz format)
            save_path = images_dir / item["aug_fname"]
            img.resize((512, 512)).save(save_path, format="JPEG", quality=92)

            # Accumulate annotation records
            new_img_id = img_id_next
            img_id_next += 1

            img_records.append({
                "id":                new_img_id,
                "file_name":         item["aug_fname"],
                "original_file_name": item["file_name"],
                "original_image_id": item["image_id"],
                "augmentation_style": item["style_key"],
            })
            ann_records.append({
                "id":                ann_id_next,
                "image_id":          new_img_id,
                "caption":           caption_index.get(item["image_id"], ""),
                "augmentation_prompt": prompts[batch.index(item)],
                "is_precanned":      False,
                "is_rejected":       False,
            })
            ann_id_next += 1

        pbar.update(len(batch))

        # Periodic checkpoint
        images_done = len(img_records)
        if images_done % args.checkpoint_every < args.batch_size:
            save_annotations(ann_records, img_records, ann_path, args.model, run_id, args.mode)
            pbar.set_postfix_str(f"checkpoint @ {images_done}")

    pbar.close()

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    save_annotations(ann_records, img_records, ann_path, args.model, run_id, args.mode)
    print(f"\nDone. {len(img_records)} augmented images saved.")
    print(f"Images      : {images_dir}")
    print(f"Annotations : {ann_path}")


if __name__ == "__main__":
    main()
