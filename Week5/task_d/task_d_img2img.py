"""
Task D – Step 2 (img2img variant): Degradation augmentation via img2img diffusion.

Loads the top-N highest-quality VizWiz training images (ranked by the
CLIP quality scores produced by clip_score.py), then generates one or
more degraded variants of each image with a Stable Diffusion img2img
pipeline. The text prompt is built from the image's reference captions.

Degradation styles (prompt templates + strength) are defined in
DEGRADATION_STYLES below and can be extended freely.

For each source image the script saves:
  * Individual augmented images   (<stem>_<style>.png)
  * A per-image comparison plot   (<stem>_comparison.png)

And at the end:
  * A combined multi-row figure   combined_augmentations.png

Usage
-----
python task_d_img2img.py

python task_d_img2img.py --n_images 10 --styles blur dark close_up --model sdxl-img2img --seed 7
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
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
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
# strength: how much the model deviates from the input (0 = no change, 1 = free gen)
# ---------------------------------------------------------------------------

DEGRADATION_STYLES = {
    "blur": {
        "prompt_template": "a blurry, out of focus version of {caption}",
        "negative":        "sharp, clear, in focus",
        "strength":        0.60,
        "label":           "Blur / out-of-focus",
    },
    "dark": {
        "prompt_template": "a dark, underexposed version of {caption}",
        "negative":        "bright, well-lit, properly exposed",
        "strength":        0.60,
        "label":           "Dark / underexposed",
    },
    "close_up": {
        "prompt_template": "an extreme close-up, partially out of frame version of {caption}",
        "negative":        "wide shot, full view, properly framed",
        "strength":        0.65,
        "label":           "Close-up / occluded",
    },
    "overexposed": {
        "prompt_template": "an overexposed, washed-out version of {caption}",
        "negative":        "properly exposed, balanced lighting",
        "strength":        0.60,
        "label":           "Overexposed / glare",
    },
    "low_quality": {
        "prompt_template": "a low quality, noisy version of {caption}",
        "negative":        "high quality, sharp, well-composed",
        "strength":        0.55,
        "label":           "Low quality / noisy",
    },
}

DEFAULT_STYLES = list(DEGRADATION_STYLES.keys())

# ---------------------------------------------------------------------------
# Model registry  (img2img pipelines)
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    # Fast – good for iteration; guidance_scale=0 is correct for turbo models
    # (conditioning is distilled in, so the prompt still steers the output)
    "sdxl-turbo-img2img": {
        "model_id":            "stabilityai/sdxl-turbo",
        "pipeline_cls":        StableDiffusionXLImg2ImgPipeline,
        "dtype":               torch.float16,
        "variant":             "fp16",
        "num_inference_steps": 4,
        "guidance_scale":      0.0,
        "img_size":            512,
    },
    # Slower but higher quality
    "sdxl-img2img": {
        "model_id":            "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline_cls":        StableDiffusionXLImg2ImgPipeline,
        "dtype":               torch.float16,
        "variant":             "fp16",
        "num_inference_steps": 30,
        "guidance_scale":      7.5,
        "img_size":            1024,
    },
    # SD-turbo (non-XL) img2img – lighter on VRAM
    "sd-turbo-img2img": {
        "model_id":            "stabilityai/sd-turbo",
        "pipeline_cls":        StableDiffusionImg2ImgPipeline,
        "dtype":               torch.float16,
        "variant":             "fp16",
        "num_inference_steps": 4,
        "guidance_scale":      0.0,
        "img_size":            512,
    },
}

DEFAULT_MODEL = "sdxl-turbo-img2img"

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
    if cfg["guidance_scale"] > 0:
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_degraded(
    pipe,
    model_key: str,
    source_img: Image.Image,
    style: dict,
    prompt: str,
    seed: int,
) -> Image.Image:
    cfg = MODEL_REGISTRY[model_key]
    src = source_img.resize((cfg["img_size"], cfg["img_size"]))
    generator = torch.Generator(device=pipe.device.type).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=style["negative"],
        image=src,
        strength=style["strength"],
        num_inference_steps=cfg["num_inference_steps"],
        guidance_scale=cfg["guidance_scale"],
        generator=generator,
    ).images[0]
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
    fig.suptitle("VizWiz training augmentations (img2img degradation)",
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
        description="Task D – img2img degradation augmentation"
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
            print(f"  -> {style_key}  [{prompt[:80]}]", flush=True)
            gen = generate_degraded(pipe, args.model, rec["orig_img"], style_obj, prompt, args.seed)
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
