"""
Task B: Exploration of inference with diffusion models.

Runs seven systematic ablations on Stable Diffusion 2.1:
  1. Scheduler     - DDPM vs DDIM at multiple step counts
  2. Prompting     - effect of positive / negative prompt engineering
  3. CFG strength  - guidance_scale sweep
  4. Denoising steps - step-count sweep (DDIM and DDPM side-by-side)
  5. Scheduler zoo - Euler / Euler-a / DPM-Solver++ / PNDM / UniPC vs DDIM
  6. DDIM eta      - stochasticity sweep (0.0 -> fully deterministic ... 1.0 -> DDPM-like)
  7. Trajectory    - denoising trajectory visualised as a filmstrip + GIF

Each experiment saves individual images + a labelled comparison grid.
A JSON summary records all parameters and timings.

Usage examples
--------------
# All experiments (default)
python explore_inference.py

# Only specific experiments
python explore_inference.py --experiments scheduler cfg steps

# Custom output dir
python explore_inference.py --output_dir /path/to/out
"""

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    DDIMScheduler,
    DDPMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SEED = 42
IMG_SIZE = 512   # height = width for SD 2.1

# Baseline prompt used across all experiments
BASE_PROMPT = (
    "a majestic red fox sitting in a snowy forest"
    "golden hour lighting, photorealistic, 8k"
)
BASE_NEG = "blurry, low quality, distorted, ugly, watermark, noise"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FONT_SIZE = 18


def _font():
    """Return a PIL font, falling back to the default if no TTF is found."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", FONT_SIZE)
    except OSError:
        return ImageFont.load_default()


def make_grid(images: list[Image.Image], labels: list[str],
              rows: int, cols: int, pad: int = 8,
              label_h: int = 28) -> Image.Image:
    """Tile images into a grid with a text label above each cell."""
    assert len(images) == rows * cols, \
        f"Expected {rows*cols} images, got {len(images)}"

    w, h = images[0].size
    cell_h = h + label_h + pad
    cell_w = w + pad
    canvas = Image.new("RGB",
                       (cols * cell_w + pad, rows * cell_h + pad),
                       color=(30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    font = _font()

    for idx, (img, lbl) in enumerate(zip(images, labels)):
        row, col = divmod(idx, cols)
        x = pad + col * cell_w
        y = pad + row * cell_h
        # Label
        draw.text((x, y), lbl, fill=(220, 220, 220), font=font)
        # Image
        canvas.paste(img, (x, y + label_h))

    return canvas

def load_pipeline(scheduler_cls, device: str):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )

    pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe
def generate(pipe, prompt, negative_prompt=None,
             num_inference_steps=25, guidance_scale=7.5,
             seed=SEED) -> tuple[Image.Image, float]:
    """Run one generation, return (image, elapsed_seconds)."""
    generator = torch.Generator(device=pipe.device.type).manual_seed(seed)
    kwargs = dict(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        height=IMG_SIZE,
        width=IMG_SIZE,
    )
    if negative_prompt is not None:
        kwargs["negative_prompt"] = negative_prompt

    t0 = time.perf_counter()
    image = pipe(**kwargs).images[0]
    elapsed = time.perf_counter() - t0
    return image, elapsed


def vram_gb(device: str) -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    return 0.0


# ---------------------------------------------------------------------------
# Experiment 1 - Scheduler: DDPM vs DDIM
# ---------------------------------------------------------------------------

SCHEDULER_STEPS = [10, 20, 50, 100]

def experiment_scheduler(output_dir: Path, device: str) -> list[dict]:
    """Compare DDPM and DDIM at several step counts (same seed & prompt)."""
    print("\n" + "=" * 60)
    print("Experiment 1 - Scheduler: DDPM vs DDIM")
    print("=" * 60)
    exp_dir = output_dir / "1_scheduler"
    exp_dir.mkdir(parents=True, exist_ok=True)

    records = []
    schedulers = {"DDIM": DDIMScheduler, "DDPM": DDPMScheduler}
    grid_images, grid_labels = [], []

    for sched_name, sched_cls in schedulers.items():
        print(f"\n  Loading pipeline with {sched_name}...")
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        pipe = load_pipeline(sched_cls, device)

        for steps in SCHEDULER_STEPS:
            print(f"    {sched_name} | steps={steps}", end="  ", flush=True)
            img, t = generate(pipe, BASE_PROMPT, BASE_NEG,
                              num_inference_steps=steps, guidance_scale=7.5)
            print(f"-> {t:.1f}s")
            fname = f"{sched_name.lower()}_{steps:03d}steps.png"
            img.save(exp_dir / fname)

            grid_images.append(img)
            grid_labels.append(f"{sched_name} / {steps} steps\n{t:.1f}s")
            records.append({
                "experiment": "scheduler",
                "scheduler": sched_name,
                "num_inference_steps": steps,
                "guidance_scale": 7.5,
                "gen_time_s": round(t, 2),
                "peak_vram_gb": round(vram_gb(device), 2),
                "file": fname,
            })

        del pipe
        torch.cuda.empty_cache()

    # Grid: 2 rows (DDIM/DDPM) x len(SCHEDULER_STEPS) cols
    grid = make_grid(grid_images, grid_labels,
                     rows=2, cols=len(SCHEDULER_STEPS))
    grid.save(exp_dir / "grid_scheduler.png")
    print(f"\n  Grid saved -> {exp_dir / 'grid_scheduler.png'}")
    return records


# ---------------------------------------------------------------------------
# Experiment 2 - Positive & Negative Prompting
# ---------------------------------------------------------------------------

PROMPT_CONDITIONS = [
    {
        "label": "bare_prompt",
        "positive": "a fox in a forest",
        "negative": None,
        "desc": "Bare prompt\n(no neg.)",
    },
    {
        "label": "detailed_pos",
        "positive": BASE_PROMPT,
        "negative": None,
        "desc": "Detailed pos.\n(no neg.)",
    },
    {
        "label": "bare_with_neg",
        "positive": "a fox in a forest",
        "negative": BASE_NEG,
        "desc": "Bare prompt\n+ neg.",
    },
    {
        "label": "full_prompts",
        "positive": BASE_PROMPT,
        "negative": BASE_NEG,
        "desc": "Detailed pos.\n+ neg.",
    },
    {
        "label": "style_tokens",
        "positive": (
            BASE_PROMPT
            + ", oil painting, impressionist style, visible brushstrokes, "
              "museum quality, trending on artstation"
        ),
        "negative": BASE_NEG + ", photo, realistic",
        "desc": "Style tokens\n(impressionist)",
    },
    {
        "label": "negative_overload",
        "positive": BASE_PROMPT,
        "negative": (
            "blurry, low quality, distorted, ugly, watermark, noise, "
            "bad anatomy, extra limbs, deformed, disfigured, "
            "cartoon, anime, painting, drawing, sketch"
        ),
        "desc": "Heavy\nnegative",
    },
]


def experiment_prompting(output_dir: Path, device: str) -> list[dict]:
    """Show impact of different positive / negative prompt combinations."""
    print("\n" + "=" * 60)
    print("Experiment 2 - Positive & Negative Prompting")
    print("=" * 60)
    exp_dir = output_dir / "2_prompting"
    exp_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    pipe = load_pipeline(DDIMScheduler, device)

    records = []
    grid_images, grid_labels = [], []

    for cond in PROMPT_CONDITIONS:
        print(f"  Condition: {cond['label']}", end="  ", flush=True)
        img, t = generate(pipe, cond["positive"], cond["negative"],
                          num_inference_steps=30, guidance_scale=7.5)
        print(f"-> {t:.1f}s")
        fname = f"{cond['label']}.png"
        img.save(exp_dir / fname)

        grid_images.append(img)
        grid_labels.append(cond["desc"] + f"\n{t:.1f}s")
        records.append({
            "experiment": "prompting",
            "label": cond["label"],
            "positive_prompt": cond["positive"],
            "negative_prompt": cond["negative"],
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "gen_time_s": round(t, 2),
            "peak_vram_gb": round(vram_gb(device), 2),
            "file": fname,
        })

    del pipe
    torch.cuda.empty_cache()

    # Single row grid
    n = len(PROMPT_CONDITIONS)
    grid = make_grid(grid_images, grid_labels, rows=1, cols=n)
    grid.save(exp_dir / "grid_prompting.png")
    print(f"\n  Grid saved -> {exp_dir / 'grid_prompting.png'}")
    return records


# ---------------------------------------------------------------------------
# Experiment 3 - CFG Guidance Scale Sweep
# ---------------------------------------------------------------------------

CFG_VALUES = [1.0, 2.0, 4.0, 6.0, 7.5, 10.0, 15.0, 20.0]


def experiment_cfg(output_dir: Path, device: str) -> list[dict]:
    """Sweep guidance_scale while keeping everything else fixed."""
    print("\n" + "=" * 60)
    print("Experiment 3 - CFG Guidance Scale Sweep")
    print("=" * 60)
    exp_dir = output_dir / "3_cfg"
    exp_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    pipe = load_pipeline(DDIMScheduler, device)

    records = []
    grid_images, grid_labels = [], []

    for cfg in CFG_VALUES:
        print(f"  CFG={cfg:.1f}", end="  ", flush=True)
        img, t = generate(pipe, BASE_PROMPT, BASE_NEG,
                          num_inference_steps=30, guidance_scale=cfg)
        print(f"-> {t:.1f}s")
        fname = f"cfg_{str(cfg).replace('.', '_')}.png"
        img.save(exp_dir / fname)

        grid_images.append(img)
        grid_labels.append(f"CFG={cfg}\n{t:.1f}s")
        records.append({
            "experiment": "cfg",
            "guidance_scale": cfg,
            "num_inference_steps": 30,
            "scheduler": "DDIM",
            "gen_time_s": round(t, 2),
            "peak_vram_gb": round(vram_gb(device), 2),
            "file": fname,
        })

    del pipe
    torch.cuda.empty_cache()

    # Single row grid
    grid = make_grid(grid_images, grid_labels, rows=1, cols=len(CFG_VALUES))
    grid.save(exp_dir / "grid_cfg.png")
    print(f"\n  Grid saved -> {exp_dir / 'grid_cfg.png'}")
    return records


# ---------------------------------------------------------------------------
# Experiment 4 - Number of Denoising Steps
# ---------------------------------------------------------------------------

STEP_VALUES = [5, 10, 15, 20, 30, 50, 75, 100]


def experiment_steps(output_dir: Path, device: str) -> list[dict]:
    """Compare output quality vs. step count for both DDIM and DDPM."""
    print("\n" + "=" * 60)
    print("Experiment 4 - Number of Denoising Steps")
    print("=" * 60)
    exp_dir = output_dir / "4_steps"
    exp_dir.mkdir(parents=True, exist_ok=True)

    records = []
    schedulers = {"DDIM": DDIMScheduler, "DDPM": DDPMScheduler}
    grid_images, grid_labels = [], []

    for sched_name, sched_cls in schedulers.items():
        print(f"\n  {sched_name}:")
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        pipe = load_pipeline(sched_cls, device)

        for steps in STEP_VALUES:
            print(f"    steps={steps:3d}", end="  ", flush=True)
            img, t = generate(pipe, BASE_PROMPT, BASE_NEG,
                              num_inference_steps=steps, guidance_scale=7.5)
            print(f"-> {t:.1f}s")
            fname = f"{sched_name.lower()}_{steps:03d}steps.png"
            img.save(exp_dir / fname)

            grid_images.append(img)
            grid_labels.append(f"{sched_name} / {steps}s\n{t:.1f}s")
            records.append({
                "experiment": "steps",
                "scheduler": sched_name,
                "num_inference_steps": steps,
                "guidance_scale": 7.5,
                "gen_time_s": round(t, 2),
                "peak_vram_gb": round(vram_gb(device), 2),
                "file": fname,
            })

        del pipe
        torch.cuda.empty_cache()

    # Grid: 2 rows (DDIM / DDPM) x len(STEP_VALUES) cols
    grid = make_grid(grid_images, grid_labels,
                     rows=2, cols=len(STEP_VALUES))
    grid.save(exp_dir / "grid_steps.png")
    print(f"\n  Grid saved -> {exp_dir / 'grid_steps.png'}")
    return records


# ---------------------------------------------------------------------------
# Experiment 5 - Scheduler Zoo
# ---------------------------------------------------------------------------

# (name, scheduler_cls, supports_karras)
SCHEDULER_ZOO = [
    ("DDIM",         DDIMScheduler,                  False),
    ("PNDM",         PNDMScheduler,                  False),
    ("Euler",        EulerDiscreteScheduler,          False),
    ("Euler-a",      EulerAncestralDiscreteScheduler, False),
    ("DPM-Solver++", DPMSolverMultistepScheduler,     False),
    ("UniPC",        UniPCMultistepScheduler,         False),
]
SCHEDULER_ZOO_STEPS = [10, 20, 50]


def experiment_scheduler_zoo(output_dir: Path, device: str) -> list[dict]:
    """Compare six modern schedulers at three step counts on one shared pipeline."""
    print("\n" + "=" * 60)
    print("Experiment 5 - Scheduler Zoo")
    print("=" * 60)
    exp_dir = output_dir / "5_scheduler_zoo"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load weights once; swap scheduler in-place for each variant
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    print("  Loading base pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    records = []
    # grid layout: rows = schedulers, cols = step counts
    grid_images = []
    grid_labels = []

    for sched_name, sched_cls, _ in SCHEDULER_ZOO:
        pipe.scheduler = sched_cls.from_config(pipe.scheduler.config)
        for steps in SCHEDULER_ZOO_STEPS:
            print(f"  {sched_name:<14} steps={steps:2d}", end="  ", flush=True)
            img, t = generate(pipe, BASE_PROMPT, BASE_NEG,
                              num_inference_steps=steps, guidance_scale=7.5)
            print(f"-> {t:.1f}s")
            fname = f"{sched_name.lower().replace('-','_').replace('+','p')}_{steps:03d}steps.png"
            img.save(exp_dir / fname)
            grid_images.append(img)
            grid_labels.append(f"{sched_name}\n{steps} steps / {t:.1f}s")
            records.append({
                "experiment": "scheduler_zoo",
                "scheduler": sched_name,
                "num_inference_steps": steps,
                "guidance_scale": 7.5,
                "gen_time_s": round(t, 2),
                "peak_vram_gb": round(vram_gb(device), 2),
                "file": fname,
            })

    del pipe
    torch.cuda.empty_cache()

    grid = make_grid(grid_images, grid_labels,
                     rows=len(SCHEDULER_ZOO), cols=len(SCHEDULER_ZOO_STEPS))
    grid.save(exp_dir / "grid_scheduler_zoo.png")
    print(f"\n  Grid saved -> {exp_dir / 'grid_scheduler_zoo.png'}")
    return records


# ---------------------------------------------------------------------------
# Experiment 6 - DDIM Eta Sweep
# ---------------------------------------------------------------------------

ETA_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def experiment_eta(output_dir: Path, device: str) -> list[dict]:
    """Sweep DDIM eta from 0 (deterministic) to 1 (DDPM-like stochastic)."""
    print("\n" + "=" * 60)
    print("Experiment 6 - DDIM Eta Sweep")
    print("=" * 60)
    exp_dir = output_dir / "6_eta"
    exp_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    pipe = load_pipeline(DDIMScheduler, device)

    records = []
    grid_images, grid_labels = [], []

    for eta in ETA_VALUES:
        print(f"  eta={eta:.1f}", end="  ", flush=True)
        generator = torch.Generator(device=device).manual_seed(SEED)
        t0 = time.perf_counter()
        image = pipe(
            prompt=BASE_PROMPT,
            negative_prompt=BASE_NEG,
            num_inference_steps=30,
            guidance_scale=7.5,
            eta=eta,
            generator=generator,
            height=IMG_SIZE,
            width=IMG_SIZE,
        ).images[0]
        t = time.perf_counter() - t0
        print(f"-> {t:.1f}s")

        fname = f"eta_{str(eta).replace('.', '_')}.png"
        image.save(exp_dir / fname)
        grid_images.append(image)
        grid_labels.append(f"eta={eta}\n{t:.1f}s")
        records.append({
            "experiment": "eta",
            "eta": eta,
            "scheduler": "DDIM",
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "gen_time_s": round(t, 2),
            "peak_vram_gb": round(vram_gb(device), 2),
            "file": fname,
        })

    del pipe
    torch.cuda.empty_cache()

    grid = make_grid(grid_images, grid_labels, rows=1, cols=len(ETA_VALUES))
    grid.save(exp_dir / "grid_eta.png")
    print(f"\n  Grid saved -> {exp_dir / 'grid_eta.png'}")
    return records


# ---------------------------------------------------------------------------
# Experiment 7 - Denoising Trajectory Visualization
# ---------------------------------------------------------------------------

TRAJECTORY_STEPS = 50       # total denoising steps
TRAJECTORY_CAPTURE = 5      # save a frame every N steps


def experiment_trajectory(output_dir: Path, device: str) -> list[dict]:
    """
    Capture and decode the latent at regular intervals during a single
    DDIM denoising run to visualise how the image forms from pure noise.
    Outputs a labelled filmstrip PNG and an animated GIF.
    """
    print("\n" + "=" * 60)
    print("Experiment 7 - Denoising Trajectory Visualization")
    print("=" * 60)
    exp_dir = output_dir / "7_trajectory"
    exp_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    pipe = load_pipeline(DDIMScheduler, device)

    snapshots: list[tuple[int, Image.Image]] = []  # (step_idx, pil_image)

    def capture_callback(step, timestep, latents):
        if step % TRAJECTORY_CAPTURE == 0 or step == TRAJECTORY_STEPS - 1:
            with torch.no_grad():
                # Decode in fp32: SDXL fp16 VAE overflows on noisy mid-step latents
                lat = latents.to(torch.float32) / pipe.vae.config.scaling_factor
                pipe.vae.to(torch.float32)
                decoded = pipe.vae.decode(lat).sample
                pipe.vae.to(latents.dtype)  # restore original dtype

            decoded = (decoded / 2 + 0.5).clamp(0, 1)

            if decoded.dim() == 3:
                decoded = decoded.unsqueeze(0)

            decoded = decoded.cpu().float()
            decoded = torch.nan_to_num(decoded, nan=0.0, posinf=1.0, neginf=0.0)

            # safe conversion
            img = decoded[0].permute(1, 2, 0).numpy()
            img = (img * 255).round().clip(0, 255).astype("uint8")

            snapshots.append((step, Image.fromarray(img)))

    print(f"  Generating {TRAJECTORY_STEPS} steps, capturing every {TRAJECTORY_CAPTURE}...")
    generator = torch.Generator(device=device).manual_seed(SEED)
    t0 = time.perf_counter()
    pipe(
        prompt=BASE_PROMPT,
        negative_prompt=BASE_NEG,
        num_inference_steps=TRAJECTORY_STEPS,
        guidance_scale=7.5,
        generator=generator,
        height=IMG_SIZE,
        width=IMG_SIZE,
        callback=capture_callback,
        callback_steps=1,
    )
    t = time.perf_counter() - t0
    print(f"  Done in {t:.1f}s  ({len(snapshots)} frames captured)")

    del pipe
    torch.cuda.empty_cache()

    # Save individual frames
    frame_imgs = []
    frame_labels = []
    for step_idx, img in snapshots:
        pct = round(step_idx / (TRAJECTORY_STEPS - 1) * 100)
        fname = f"step_{step_idx:03d}.png"
        img.save(exp_dir / fname)
        frame_imgs.append(img)
        frame_labels.append(f"step {step_idx}\n({pct}%)")

    # Filmstrip grid (1 row)
    grid = make_grid(frame_imgs, frame_labels, rows=1, cols=len(frame_imgs))
    grid.save(exp_dir / "filmstrip.png")
    print(f"  Filmstrip -> {exp_dir / 'filmstrip.png'}")

    # Animated GIF
    gif_path = exp_dir / "trajectory.gif"
    frame_imgs[0].save(
        gif_path,
        save_all=True,
        append_images=frame_imgs[1:],
        duration=200,    # ms per frame
        loop=0,
    )
    print(f"  GIF       -> {gif_path}")

    return [{
        "experiment": "trajectory",
        "scheduler": "DDIM",
        "num_inference_steps": TRAJECTORY_STEPS,
        "capture_every_n_steps": TRAJECTORY_CAPTURE,
        "frames_captured": len(snapshots),
        "guidance_scale": 7.5,
        "gen_time_s": round(t, 2),
        "peak_vram_gb": round(vram_gb(device), 2),
    }]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXPERIMENT_MAP = {
    "scheduler":      experiment_scheduler,
    "prompting":      experiment_prompting,
    "cfg":            experiment_cfg,
    "steps":          experiment_steps,
    "scheduler_zoo":  experiment_scheduler_zoo,
    "eta":            experiment_eta,
    "trajectory":     experiment_trajectory,
}


def main():
    parser = argparse.ArgumentParser(description="Task B - Inference exploration")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENT_MAP.keys()),
        choices=list(EXPERIMENT_MAP.keys()),
        help="Which experiments to run (default: all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/ghome/group07/MCV-C5-Group7/Week5/task_b/output_results",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"VRAM   : {total_vram:.1f} GB")
    print(f"Model  : {MODEL_ID}")

    all_records = []
    for name in args.experiments:
        fn = EXPERIMENT_MAP[name]
        records = fn(output_dir, device)
        all_records.extend(records)

    summary_path = output_dir / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nFull summary -> {summary_path}")

    # Quick text table
    print("\n" + "=" * 72)
    print(f"{'Experiment':<12} {'Details':<32} {'Steps':>6} {'CFG':>5} {'Time(s)':>8}")
    print("-" * 72)
    for r in all_records:
        exp = r["experiment"]
        if exp == "scheduler":
            detail = f"sched={r['scheduler']}"
        elif exp == "prompting":
            detail = r["label"]
        elif exp == "cfg":
            detail = f"cfg={r['guidance_scale']}"
        elif exp == "scheduler_zoo":
            detail = f"sched={r['scheduler']}"
        elif exp == "eta":
            detail = f"eta={r['eta']}"
        elif exp == "trajectory":
            detail = f"{r['frames_captured']} frames"
        else:  # steps
            detail = f"sched={r['scheduler']}"
        steps_val = r.get("num_inference_steps", r.get("frames_captured", 0))
        cfg_val   = r.get("guidance_scale", 0.0)
        print(
            f"{exp:<12} {detail:<32} {steps_val:>6} "
            f"{cfg_val:>5.1f} {r['gen_time_s']:>8.2f}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
