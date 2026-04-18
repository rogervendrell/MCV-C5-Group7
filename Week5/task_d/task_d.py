"""
Task D – Step 2: Degradation augmentation via text-to-image diffusion.

Loads the top-N highest-quality VizWiz training images (ranked by the
CLIP quality scores produced by clip_score.py), then generates degraded
variants purely from text prompts — no image conditioning.

The prompt is built from the image's reference caption:
    "a blurry, out of focus version of <caption>"
    "a dark, underexposed version of <caption>"
    etc.

This gives the model full freedom to vary the output while still being
semantically grounded in what the original image shows.

For each source image the script saves:
  * Individual generated images    (<stem>_<style>.png)
  * A per-image comparison plot    (<stem>_comparison.png)
    (original image shown on the left for visual reference)

And at the end:
  * A combined multi-row figure    combined_augmentations.png

Usage
-----
# Defaults: top-5 images, all degradation styles, sdxl model
python task_d.py

# Custom
python task_d.py --n_images 10 --styles blur dark close_up --model sdxl --seed 7
"""

import argparse
import textwrap
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DDIMScheduler,
)
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

VIZWIZ_TRAIN_IMG_DIR = Path("/ghome/group07/MCV-C5-Group7/Week3/dataset/train")
VIZWIZ_TRAIN_ANN     = Path("/ghome/group07/MCV-C5-Group7/Week3/dataset/annotations/train.json")
DEFAULT_SCORES_JSON  = Path("/ghome/group07/MCV-C5-Group7/Week5/task_d/results/clip_scores.json")
DEFAULT_OUTPUT_DIR   = Path("/ghome/group07/MCV-C5-Group7/Week5/task_d/results")

# ---------------------------------------------------------------------------
# Degradation style definitions
# ---------------------------------------------------------------------------

# Shared negative suffix: explicitly forbid the high-quality defaults SDXL drifts toward
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

DEFAULT_STYLES = list(DEGRADATION_STYLES.keys())

# ---------------------------------------------------------------------------
# Model registry  (text-to-image pipelines)
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

DEFAULT_MODEL = "sdxl"

# ---------------------------------------------------------------------------
# Helpers – data loading
# ---------------------------------------------------------------------------

def load_top_n(scores_json: Path, n: int) -> list[dict]:
    with open(scores_json) as f:
        data = json.load(f)
    return data["scores"][:n]


def load_caption_index(ann_file: Path) -> dict[int, str]:
    from collections import defaultdict
    with open(ann_file) as f:
        data = json.load(f)
    captions: dict[int, list[str]] = defaultdict(list)
    for ann in data["annotations"]:
        if not ann.get("is_rejected", False):
            captions[ann["image_id"]].append(ann["caption"].strip())
    return {img_id: caps[0] for img_id, caps in captions.items() if caps}


def build_prompt(style: dict, caption: str) -> str:
    return style["prompt_template"].format(caption=caption)


# ---------------------------------------------------------------------------
# Helpers – generation
# ---------------------------------------------------------------------------

def load_pipeline(model_key: str, device: str):
    cfg = MODEL_REGISTRY[model_key]
    print(f"  Loading {model_key} ({cfg['model_id']}) ...", flush=True)
    pipe = cfg["pipeline_cls"].from_pretrained(
        cfg["model_id"],
        torch_dtype=cfg["dtype"],
        use_safetensors=True,
        variant=cfg["variant"],
    ).to(device)
    # Only swap to DDIM for full models; turbo models use their own scheduler
    if cfg["guidance_scale"] > 0:
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate(pipe, model_key: str, prompt: str, negative: str, seed: int) -> Image.Image:
    cfg = MODEL_REGISTRY[model_key]
    generator = torch.Generator(device=pipe.device.type).manual_seed(seed)
    kwargs = dict(
        prompt=prompt,
        num_inference_steps=cfg["num_inference_steps"],
        guidance_scale=cfg["guidance_scale"],
        generator=generator,
        height=cfg["img_size"],
        width=cfg["img_size"],
    )
    if cfg["guidance_scale"] > 0:
        kwargs["negative_prompt"] = negative
    result = pipe(**kwargs).images[0]
    return result.resize((512, 512))


# ---------------------------------------------------------------------------
# Helpers – plotting
# ---------------------------------------------------------------------------

def _wrap(text: str, width: int = 26) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def plot_row(axes, orig_img, gen_imgs, style_labels, file_name):
    axes[0].imshow(orig_img)
    axes[0].set_title(f"Original\n{file_name}", fontsize=7, fontweight="bold", pad=3)
    axes[0].axis("off")
    for ax, img, lbl in zip(axes[1:], gen_imgs, style_labels):
        ax.imshow(img)
        ax.set_title(_wrap(lbl), fontsize=7, pad=3)
        ax.axis("off")


def save_individual_plot(orig_img, gen_imgs, style_labels, file_name, output_path):
    n_cols = 1 + len(gen_imgs)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.8), squeeze=False)
    plot_row(axes[0], orig_img, gen_imgs, style_labels, file_name)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_combined_plot(records, style_labels, output_path):
    n_rows = len(records)
    n_cols = 1 + len(style_labels)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols, 3.5 * n_rows), squeeze=False)
    for row_idx, rec in enumerate(records):
        plot_row(axes[row_idx], rec["orig_img"], rec["gen_imgs"], style_labels, rec["file_name"])
    fig.suptitle("VizWiz training augmentations (txt2img degradation)",
                 fontsize=11, fontweight="bold", y=1.005)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Combined figure -> {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Task D – txt2img degradation augmentation"
    )
    parser.add_argument("--n_images",    type=int, default=5)
    parser.add_argument("--styles",      nargs="+", default=DEFAULT_STYLES,
                        choices=list(DEGRADATION_STYLES.keys()))
    parser.add_argument("--model",       default=DEFAULT_MODEL,
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--scores_json", type=str, default=str(DEFAULT_SCORES_JSON))
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--output_dir",  type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device     : {device}")
    if torch.cuda.is_available():
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"Model      : {args.model}")
    print(f"N images   : {args.n_images}")
    print(f"Styles     : {args.styles}")
    print()

    scores_path = Path(args.scores_json)
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores file not found: {scores_path}\n"
                                "Run clip_score.py first.")

    top_entries = load_top_n(scores_path, args.n_images)
    print(f"Loaded top-{len(top_entries)} images")
    for e in top_entries:
        print(f"  {e['file_name']:42s}  quality={e['quality_score']:+.4f}")
    print()

    print("Loading caption index ...")
    caption_index = load_caption_index(VIZWIZ_TRAIN_ANN)

    records = []
    for entry in top_entries:
        orig = Image.open(VIZWIZ_TRAIN_IMG_DIR / entry["file_name"]).convert("RGB").resize((512, 512))
        records.append({
            "image_id":      entry["image_id"],
            "file_name":     entry["file_name"],
            "quality_score": entry["quality_score"],
            "caption":       caption_index.get(entry["image_id"], "a photograph"),
            "orig_img":      orig,
            "gen_imgs":      [],
        })

    style_objects = [DEGRADATION_STYLES[s] for s in args.styles]
    style_labels  = [DEGRADATION_STYLES[s]["label"] for s in args.styles]

    print(f"Loading pipeline: {args.model}")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    pipe = load_pipeline(args.model, device)

    for i, rec in enumerate(records):
        print(f"\n[{i+1}/{len(records)}] {rec['file_name']}  (quality={rec['quality_score']:+.4f})")
        stem = Path(rec["file_name"]).stem

        for style_key, style_obj in zip(args.styles, style_objects):
            prompt = build_prompt(style_obj, rec["caption"])
            print(f"  -> {style_key}  [{prompt}]", flush=True)
            gen = generate(pipe, args.model, prompt, style_obj["negative"], args.seed)
            rec["gen_imgs"].append(gen)
            gen.save(output_dir / f"{stem}_{style_key}.png")

        save_individual_plot(rec["orig_img"], rec["gen_imgs"], style_labels,
                             rec["file_name"], output_dir / f"{stem}_comparison.png")

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    save_combined_plot(records, style_labels, output_dir / "combined_augmentations.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
