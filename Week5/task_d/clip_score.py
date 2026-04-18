"""
Task D – Step 1: CLIP-based image quality scoring on the VizWiz training set.

Uses CLIP to compare every training image against a positive prompt
("clear, sharp, well-lit photograph") and a negative prompt
("blurry, dark, poorly-lit photograph").

Quality score = cosine_sim(image, positive) - cosine_sim(image, negative)

The result is written to a JSON file so that any downstream script can
load it and select the top-N images without re-running CLIP.

Features
--------
* Batched GPU inference (configurable --batch_size)
* Checkpoint / resume: already-scored images are skipped if the output
  JSON already exists, so an interrupted job can be safely re-submitted.
* Progress bar via tqdm.

Usage
-----
# Score all training images (default settings)
python clip_score.py

# Custom batch size / output path
python clip_score.py --batch_size 128 --output_json /path/to/clip_scores.json
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

VIZWIZ_TRAIN_IMG_DIR = Path("/ghome/group07/MCV-C5-Group7/Week3/dataset/train")
VIZWIZ_TRAIN_ANN     = Path("/ghome/group07/MCV-C5-Group7/Week3/dataset/annotations/train.json")

DEFAULT_OUTPUT_JSON  = Path("/ghome/group07/MCV-C5-Group7/Week5/task_d/results/clip_scores.json")

# ---------------------------------------------------------------------------
# CLIP model
# ---------------------------------------------------------------------------

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"

# ---------------------------------------------------------------------------
# Quality prompts
# ---------------------------------------------------------------------------

POSITIVE_PROMPTS = [
    "a clear, sharp, well-lit photograph with good focus and proper framing",
    "a high quality photo with good lighting and sharp focus",
    "a well-composed, properly exposed photograph",
]

NEGATIVE_PROMPTS = [
    "a blurry, out of focus, dark, poorly lit photograph",
    "a low quality photo that is blurry or overexposed or poorly framed",
    "a bad photograph with motion blur, bad lighting, or severe occlusion",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_train_index(ann_file: Path) -> list[dict]:
    """Return list of {image_id, file_name} for the training set."""
    with open(ann_file) as f:
        data = json.load(f)
    return [{"image_id": img["id"], "file_name": img["file_name"]}
            for img in data["images"]]


def encode_texts(model, processor, texts: list[str], device: str) -> torch.Tensor:
    """Return L2-normalised text embeddings, shape (len(texts), D).

    Goes through model.text_model + model.text_projection directly to avoid
    a transformers version issue where get_text_features() returns a
    BaseModelOutputWithPooling object instead of a plain tensor.
    """
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_out = model.text_model(**inputs)
        feats = model.text_projection(text_out.pooler_output)
    return F.normalize(feats, dim=-1)


def encode_images_batch(model, processor, image_paths: list[Path], device: str) -> torch.Tensor:
    """Return L2-normalised image embeddings for a batch, shape (B, D).

    Goes through model.vision_model + model.visual_projection directly for
    the same reason as encode_texts.
    """
    images = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            # Fallback: black 224×224 image for corrupted files
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        images.append(img)

    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        vision_out = model.vision_model(**inputs)
        feats = model.visual_projection(vision_out.pooler_output)
    return F.normalize(feats, dim=-1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Task D – Step 1: CLIP quality scoring of VizWiz training images"
    )
    parser.add_argument("--batch_size",  type=int,  default=64,
                        help="Images per CLIP forward pass (default: 64)")
    parser.add_argument("--output_json", type=str,
                        default=str(DEFAULT_OUTPUT_JSON),
                        help="Where to write the scores JSON")
    args = parser.parse_args()

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device     : {device}")
    if torch.cuda.is_available():
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"CLIP model : {CLIP_MODEL_ID}")
    print(f"Batch size : {args.batch_size}")
    print(f"Output     : {output_path}")
    print()

    # ------------------------------------------------------------------
    # Load existing results for checkpoint / resume
    # ------------------------------------------------------------------
    existing: dict[str, dict] = {}
    if output_path.exists():
        with open(output_path) as f:
            existing_list = json.load(f).get("scores", [])
        existing = {r["file_name"]: r for r in existing_list}
        print(f"Resuming: {len(existing)} images already scored.")

    # ------------------------------------------------------------------
    # Load CLIP
    # ------------------------------------------------------------------
    print("Loading CLIP model ...")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model     = CLIPModel.from_pretrained(CLIP_MODEL_ID, torch_dtype=torch.float16).to(device)
    model.eval()

    # ------------------------------------------------------------------
    # Encode text prompts (once)
    # ------------------------------------------------------------------
    pos_embs = encode_texts(model, processor, POSITIVE_PROMPTS, device)  # (P, D)
    neg_embs = encode_texts(model, processor, NEGATIVE_PROMPTS, device)  # (N, D)
    # Average across multiple prompts for robustness
    pos_emb = pos_embs.mean(dim=0, keepdim=True)   # (1, D)
    neg_emb = neg_embs.mean(dim=0, keepdim=True)   # (1, D)
    pos_emb = F.normalize(pos_emb, dim=-1)
    neg_emb = F.normalize(neg_emb, dim=-1)

    # ------------------------------------------------------------------
    # Load training index
    # ------------------------------------------------------------------
    train_index = load_train_index(VIZWIZ_TRAIN_ANN)
    print(f"Training images in annotation file: {len(train_index)}")

    # Filter out already-scored entries
    to_score = [e for e in train_index if e["file_name"] not in existing]
    print(f"Images to score: {len(to_score)}\n")

    # ------------------------------------------------------------------
    # Batch-score images
    # ------------------------------------------------------------------
    new_results: list[dict] = []

    pbar = tqdm(
        total=len(to_score),
        desc="Scoring images",
        unit="img",
        dynamic_ncols=True,
    )

    for batch_start in range(0, len(to_score), args.batch_size):
        batch_entries = to_score[batch_start: batch_start + args.batch_size]
        batch_paths   = [VIZWIZ_TRAIN_IMG_DIR / e["file_name"] for e in batch_entries]

        img_embs = encode_images_batch(model, processor, batch_paths, device)  # (B, D)

        pos_sims = (img_embs @ pos_emb.T).squeeze(1).cpu().float().tolist()  # (B,)
        neg_sims = (img_embs @ neg_emb.T).squeeze(1).cpu().float().tolist()  # (B,)

        for entry, pos_s, neg_s in zip(batch_entries, pos_sims, neg_sims):
            new_results.append({
                "image_id":      entry["image_id"],
                "file_name":     entry["file_name"],
                "pos_score":     round(pos_s, 6),
                "neg_score":     round(neg_s, 6),
                "quality_score": round(pos_s - neg_s, 6),
            })

        pbar.update(len(batch_entries))

    pbar.close()

    # ------------------------------------------------------------------
    # Merge with existing, sort by quality score descending, save
    # ------------------------------------------------------------------
    all_results = list(existing.values()) + new_results
    all_results.sort(key=lambda r: r["quality_score"], reverse=True)

    output = {
        "clip_model":       CLIP_MODEL_ID,
        "positive_prompts": POSITIVE_PROMPTS,
        "negative_prompts": NEGATIVE_PROMPTS,
        "total_images":     len(all_results),
        "scores":           all_results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(all_results)} scores -> {output_path}")
    print("\nTop-10 highest quality images:")
    for r in all_results[:10]:
        print(f"  {r['file_name']:40s}  quality={r['quality_score']:+.4f}  "
              f"pos={r['pos_score']:.4f}  neg={r['neg_score']:.4f}")

    print("\nBottom-10 lowest quality images:")
    for r in all_results[-10:]:
        print(f"  {r['file_name']:40s}  quality={r['quality_score']:+.4f}  "
              f"pos={r['pos_score']:.4f}  neg={r['neg_score']:.4f}")


if __name__ == "__main__":
    main()
