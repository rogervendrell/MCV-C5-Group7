"""
Task C: VizWiz caption-conditioned image generation.

For N randomly sampled VizWiz images:
  1. Load the original image and its reference captions.
  2. Combine the captions into a single generation prompt.
  3. Generate a similar image with one or more Stable Diffusion models.
  4. Save a per-image subplot (original | captions | generated x model)
     and a combined multi-image figure.

Usage
-----
# Default: 5 images, models sd-turbo + sdxl-turbo
python task_c.py

# Custom
python task_c.py --n_images 10 --models sd-turbo sdxl sdxl-turbo --seed 0

# Specific output dir
python task_c.py --output_dir /path/to/out
"""

import argparse
import json
import random
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    AutoPipelineForText2Image,
)

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------

VIZWIZ_IMG_DIR   = Path("/ghome/group07/MCV-C5-Group7/Week3/dataset/val")
VIZWIZ_ANN_FILE  = Path("/ghome/group07/MCV-C5-Group7/Week3/dataset/annotations/val.json")

# ---------------------------------------------------------------------------
# Model registry  (same as task_a for consistency)
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "sd-2-1": {
        "model_id":            "stabilityai/stable-diffusion-2-1",
        "pipeline_cls":        StableDiffusionPipeline,
        "dtype":               torch.float16,
        "num_inference_steps": 25,
        "guidance_scale":      7.5,
    },
    "sd-turbo": {
        "model_id":            "stabilityai/sd-turbo",
        "pipeline_cls":        AutoPipelineForText2Image,
        "dtype":               torch.float16,
        "num_inference_steps": 4,
        "guidance_scale":      0.0,
    },
    "sdxl": {
        "model_id":            "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline_cls":        StableDiffusionXLPipeline,
        "dtype":               torch.float16,
        "num_inference_steps": 25,
        "guidance_scale":      7.5,
    },
    "sdxl-turbo": {
        "model_id":            "stabilityai/sdxl-turbo",
        "pipeline_cls":        AutoPipelineForText2Image,
        "dtype":               torch.float16,
        "num_inference_steps": 4,
        "guidance_scale":      0.0,
    },
    "sd-3-5-medium": {
        "model_id":            "stabilityai/stable-diffusion-3.5-medium",
        "pipeline_cls":        AutoPipelineForText2Image,
        "dtype":               torch.bfloat16,
        "num_inference_steps": 25,
        "guidance_scale":      4.5,
    },
}

DEFAULT_MODELS = ["sd-turbo", "sdxl-turbo"]

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_vizwiz_index(ann_file: Path):
    """Return {image_id: {file_name, captions: [str]}}."""
    with open(ann_file) as f:
        data = json.load(f)

    # Build image_id -> file_name map
    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

    # Group captions by image_id (skip rejected ones)
    captions_by_id = defaultdict(list)
    for ann in data["annotations"]:
        if not ann.get("is_rejected", False):
            captions_by_id[ann["image_id"]].append(ann["caption"])

    index = {}
    for img_id, file_name in id_to_file.items():
        caps = captions_by_id.get(img_id, [])
        if caps:
            index[img_id] = {"file_name": file_name, "captions": caps}

    return index


def build_prompt(captions: list[str]) -> str:
    """
    Merge up to 5 reference captions into one generation prompt.
    Duplicate, whitespace-normalised sentences are de-duplicated.
    """
    seen, unique = set(), []
    for cap in captions:
        cap = cap.strip().rstrip(".")
        low = cap.lower()
        if low not in seen:
            seen.add(low)
            unique.append(cap)
    return ". ".join(unique[:5]) + "."


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def load_pipeline(model_key: str, device: str):
    cfg = MODEL_REGISTRY[model_key]
    print(f"  Loading {model_key} ({cfg['model_id']}) ...", flush=True)
    pipe = cfg["pipeline_cls"].from_pretrained(
        cfg["model_id"],
        torch_dtype=cfg["dtype"],
        use_safetensors=True,
        variant="fp16" if cfg["dtype"] == torch.float16 else None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_image(pipe, model_key: str, prompt: str, seed: int) -> Image.Image:
    cfg = MODEL_REGISTRY[model_key]
    generator = torch.Generator(device=pipe.device.type).manual_seed(seed)
    kwargs = dict(
        prompt=prompt,
        num_inference_steps=cfg["num_inference_steps"],
        guidance_scale=cfg["guidance_scale"],
        generator=generator,
        height=512,
        width=512,
    )
    return pipe(**kwargs).images[0]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _wrap(text: str, width: int = 42) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def plot_row(ax_orig, ax_text, ax_gens: list,
             orig_img: Image.Image,
             captions: list[str],
             gen_imgs: list[Image.Image],
             model_names: list[str],
             img_label: str):
    """Fill one row of subplots for a single VizWiz image."""

    # --- original image ---
    ax_orig.imshow(orig_img)
    ax_orig.set_title(img_label, fontsize=8, fontweight="bold", pad=3)
    ax_orig.axis("off")

    # --- captions text panel ---
    ax_text.axis("off")
    caption_text = "\n\n".join(
        f"[{i+1}] {_wrap(cap)}" for i, cap in enumerate(captions)
    )
    ax_text.text(
        0.05, 0.95, caption_text,
        transform=ax_text.transAxes,
        va="top", ha="left",
        fontsize=6.5,
        family="monospace",
        wrap=True,
    )
    ax_text.set_title("Reference captions", fontsize=8, pad=3)

    # --- generated images ---
    for ax, img, model_name in zip(ax_gens, gen_imgs, model_names):
        ax.imshow(img)
        ax.set_title(model_name, fontsize=8, pad=3)
        ax.axis("off")


def make_combined_figure(
    records: list[dict],
    model_names: list[str],
    output_path: Path,
):
    """
    Build a multi-row figure: one row per VizWiz image.
    Columns: [original] [captions] [model_0 gen] [model_1 gen] ...
    """
    n_images = len(records)
    n_cols   = 2 + len(model_names)   # orig + text + one per model

    fig_w = 3.5 * n_cols
    fig_h = 3.5 * n_images

    fig, axes = plt.subplots(
        n_images, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    for row_idx, rec in enumerate(records):
        ax_row = axes[row_idx]
        ax_orig = ax_row[0]
        ax_text = ax_row[1]
        ax_gens  = list(ax_row[2:])

        plot_row(
            ax_orig, ax_text, ax_gens,
            orig_img    = rec["orig_img"],
            captions    = rec["captions"],
            gen_imgs    = rec["gen_imgs"],
            model_names = model_names,
            img_label   = rec["file_name"],
        )

    fig.suptitle(
        "VizWiz → caption-conditioned generation",
        fontsize=11, fontweight="bold", y=1.005,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Combined figure saved -> {output_path}")


def make_individual_figure(rec: dict, model_names: list[str], output_path: Path):
    """Save a standalone figure for a single VizWiz sample."""
    n_cols = 2 + len(model_names)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.8), squeeze=False)
    ax_row = axes[0]

    plot_row(
        ax_orig     = ax_row[0],
        ax_text     = ax_row[1],
        ax_gens     = list(ax_row[2:]),
        orig_img    = rec["orig_img"],
        captions    = rec["captions"],
        gen_imgs    = rec["gen_imgs"],
        model_names = model_names,
        img_label   = rec["file_name"],
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Task C – VizWiz caption-conditioned generation"
    )
    parser.add_argument(
        "--n_images", type=int, default=5,
        help="Number of random VizWiz images to process (default: 5)",
    )
    parser.add_argument(
        "--models", nargs="+",
        default=DEFAULT_MODELS,
        choices=list(MODEL_REGISTRY.keys()),
        help=f"Models to use (default: {DEFAULT_MODELS})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for image sampling and generation (default: 42)",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/ghome/group07/MCV-C5-Group7/Week5/task_c/results",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"Models : {args.models}")
    print(f"N images: {args.n_images}")
    print(f"Seed   : {args.seed}")
    print()

    # ------------------------------------------------------------------
    # 1. Sample N random VizWiz images
    # ------------------------------------------------------------------
    print("Loading VizWiz index ...")
    index = load_vizwiz_index(VIZWIZ_ANN_FILE)
    all_ids = list(index.keys())

    rng = random.Random(args.seed)
    chosen_ids = rng.sample(all_ids, k=min(args.n_images, len(all_ids)))
    print(f"Sampled {len(chosen_ids)} images from {len(all_ids)} available.\n")

    # ------------------------------------------------------------------
    # 2. Build per-image records (load originals + prompts)
    # ------------------------------------------------------------------
    records = []
    for img_id in chosen_ids:
        entry     = index[img_id]
        img_path  = VIZWIZ_IMG_DIR / entry["file_name"]
        orig_img  = Image.open(img_path).convert("RGB").resize((512, 512))
        captions  = entry["captions"]
        prompt    = build_prompt(captions)

        records.append({
            "img_id":    img_id,
            "file_name": entry["file_name"],
            "orig_img":  orig_img,
            "captions":  captions,
            "prompt":    prompt,
            "gen_imgs":  [],           # filled in below
        })

    # ------------------------------------------------------------------
    # 3. Generate images — load each model once, run all N images
    # ------------------------------------------------------------------
    for model_key in args.models:
        print(f"\n{'='*60}")
        print(f"Generating with: {model_key}")
        print(f"{'='*60}")

        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        pipe = load_pipeline(model_key, device)

        for i, rec in enumerate(records):
            print(f"  [{i+1}/{len(records)}] {rec['file_name']}", flush=True)
            print(f"    Prompt: {rec['prompt'][:100]}...")
            gen = generate_image(pipe, model_key, rec["prompt"], seed=args.seed)
            rec["gen_imgs"].append(gen)

            # Save individual generated image
            stem = Path(rec["file_name"]).stem
            gen.save(output_dir / f"{stem}_{model_key}.png")

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 4. Save individual per-image plots
    # ------------------------------------------------------------------
    print("\nSaving individual plots ...")
    for rec in records:
        stem = Path(rec["file_name"]).stem
        make_individual_figure(
            rec,
            model_names=args.models,
            output_path=output_dir / f"{stem}_comparison.png",
        )

    # ------------------------------------------------------------------
    # 5. Save combined figure (all N images in one plot)
    # ------------------------------------------------------------------
    print("Saving combined figure ...")
    make_combined_figure(
        records,
        model_names=args.models,
        output_path=output_dir / "combined_vizwiz_generation.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
