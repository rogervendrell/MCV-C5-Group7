"""
Task A: Compare Stable Diffusion models.
Generates images from the same prompt with 4-5 models,
recording inference time and peak GPU memory per model.
"""

import argparse
import time
import json
import os
from pathlib import Path

import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    AutoPipelineForText2Image,
)
from PIL import Image
import math

# Models
MODELS = {
    "sd-2-1": {
        "model_id": "stabilityai/stable-diffusion-2-1",
        "pipeline_cls": StableDiffusionPipeline,
        "dtype": torch.float16,
        "num_inference_steps": 25,
        "guidance_scale": 7.5,
    },
    "sd-turbo": {
        "model_id": "stabilityai/sd-turbo",
        "pipeline_cls": AutoPipelineForText2Image,
        "dtype": torch.float16,
        "num_inference_steps": 1,   # turbo is designed for 1-4 steps
        "guidance_scale": 0.0,      # CFG-free
    },
    "sdxl": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline_cls": StableDiffusionXLPipeline,
        "dtype": torch.float16,
        "num_inference_steps": 25,
        "guidance_scale": 7.5,
    },
    "sdxl-turbo": {
        "model_id": "stabilityai/sdxl-turbo",
        "pipeline_cls": AutoPipelineForText2Image,
        "dtype": torch.float16,
        "num_inference_steps": 1,
        "guidance_scale": 0.0,
    },
    "sd-3-5-medium": {
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "pipeline_cls": AutoPipelineForText2Image,
        "dtype": torch.bfloat16,
        "num_inference_steps": 25,
        "guidance_scale": 4.5,
    },
}

PROMPT = (
    "a majestic red fox sitting in a snowy forest"
    "golden hour lighting, photorealistic, 8k"
)
NEGATIVE_PROMPT = "blurry, low quality, distorted, ugly"

SEED = 42


def load_and_generate(model_key: str, cfg: dict, output_dir: Path, device: str):
    """Load a pipeline, generate one image, return timing and memory stats."""
    print(f"\n{'='*60}")
    print(f"Model: {model_key}  ({cfg['model_id']})")
    print(f"{'='*60}")

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    t_load_start = time.perf_counter()
    pipe = cfg["pipeline_cls"].from_pretrained(
        cfg["model_id"],
        torch_dtype=cfg["dtype"],
        use_safetensors=True,
        variant="fp16" if cfg["dtype"] == torch.float16 else None,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    t_load_end = time.perf_counter()
    load_time = t_load_end - t_load_start
    print(f"  Load time : {load_time:.1f}s")

    generator = torch.Generator(device=device).manual_seed(SEED)

    # Build call kwargs — SD-turbo / SDXL-turbo don't accept negative_prompt
    call_kwargs = dict(
        prompt=PROMPT,
        num_inference_steps=cfg["num_inference_steps"],
        guidance_scale=cfg["guidance_scale"],
        generator=generator,
    )
    if cfg["guidance_scale"] > 0:
        call_kwargs["negative_prompt"] = NEGATIVE_PROMPT

    t_gen_start = time.perf_counter()
    image = pipe(**call_kwargs).images[0]
    t_gen_end = time.perf_counter()
    gen_time = t_gen_end - t_gen_start
    print(f"  Gen time  : {gen_time:.1f}s  ({cfg['num_inference_steps']} steps)")

    peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(f"  Peak VRAM : {peak_mem_gb:.2f} GB")

    out_path = output_dir / f"{model_key}.png"
    image.save(out_path)
    print(f"  Saved     : {out_path}")

    # Free memory before next model
    del pipe
    torch.cuda.empty_cache()

    return {
        "model_key": model_key,
        "model_id": cfg["model_id"],
        "num_inference_steps": cfg["num_inference_steps"],
        "guidance_scale": cfg["guidance_scale"],
        "load_time_s": round(load_time, 2),
        "gen_time_s": round(gen_time, 2),
        "peak_vram_gb": round(peak_mem_gb, 2),
    }

def create_image_grid(image_paths, output_path, cols=3):
    """Create a grid image from a list of image paths."""
    images = [Image.open(p) for p in image_paths]

    # Assume all same size
    w, h = images[0].size

    rows = math.ceil(len(images) / cols)

    grid = Image.new("RGB", (cols * w, rows * h))

    for i, img in enumerate(images):
        x = (i % cols) * w
        y = (i // cols) * h
        grid.paste(img, (x, y))

    grid.save(output_path)
    print(f"\nGrid saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Compare SD models (Task A)")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
        help="Which models to run (default: all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/ghome/group07/MCV-C5-Group7/Week5/TaskA/output",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"VRAM  : {total_vram:.1f} GB")

    results = []
    image_paths = []
    for key in args.models:
        cfg = MODELS[key]
        stats = load_and_generate(key, cfg, output_dir, device)
        results.append(stats)
        image_paths.append(output_dir / f"{key}.png")

    # Save summary JSON
    summary_path = output_dir / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Print table
    print("\n" + "="*75)
    print(f"{'Model':<18} {'Steps':>6} {'CFG':>5} {'Load(s)':>9} {'Gen(s)':>8} {'VRAM(GB)':>10}")
    print("-"*75)
    for r in results:
        print(
            f"{r['model_key']:<18} {r['num_inference_steps']:>6} "
            f"{r['guidance_scale']:>5.1f} {r['load_time_s']:>9.1f} "
            f"{r['gen_time_s']:>8.1f} {r['peak_vram_gb']:>10.2f}"
        )
    print("="*75)

    grid_path = output_dir / "grid.png"
    create_image_grid(image_paths, grid_path, cols=3)


if __name__ == "__main__":
    main()
