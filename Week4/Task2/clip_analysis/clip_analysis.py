"""
CLIP-based caption–image alignment analysis for Task 2 predictions.

For every validation sample we compute:
  clip_gen  = cosine_sim(image, generated_caption)
  clip_gt   = max cosine_sim(image, reference_i)   over all references
  bleu1     = sentence BLEU-1 (generated vs references, smoothed)

The key question: are there cases where clip_gen is HIGH but BLEU is LOW?
Those are samples where the model is semantically correct but penalised by
n-gram mismatch — i.e. BLEU under-rates actual quality.

Usage
-----
    # Run from any directory (paths are hardcoded to the Task2 tree):
    python clip_analysis.py

    # Skip recomputing scores if cache already exists:
    python clip_analysis.py --use-cache

    # Choose a different predictions file:
    python clip_analysis.py --predictions /path/to/predictions_val.json

Outputs (all saved to the same directory as this script)
---------
    scores.json          — per-sample clip_gen, clip_gt, bleu1
    scatter_clip_vs_bleu.png
    distributions.png
    scatter_gen_vs_gt.png
    hidden_gems.png      — high CLIP / low BLEU gallery
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE             = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PREDS     = (
    '/ghome/group07/MCV-C5-Group7/Week4/Task2/results'
    '/qwen_baseline_2b/109572/predictions_val.json'
)
VAL_IMAGE_DIR     = '/ghome/group07/MCV-C5-Group7/Week3/dataset/val'
CLIP_MODEL_NAME   = 'openai/clip-vit-base-patch32'
SCORES_CACHE      = os.path.join(_HERE, 'scores.json')

BATCH_SIZE        = 64   # images + texts per forward pass
MAX_TEXT_LENGTH   = 77   # CLIP hard limit

# ---------------------------------------------------------------------------
# BLEU helper
# ---------------------------------------------------------------------------
_smoother = SmoothingFunction().method1

def bleu1(prediction: str, references: list[str]) -> float:
    hyp  = prediction.strip().split()
    refs = [r.strip().split() for r in references if r.strip()]
    if not hyp or not refs:
        return 0.0
    return sentence_bleu(refs, hyp, weights=(1, 0, 0, 0),
                         smoothing_function=_smoother)


# ---------------------------------------------------------------------------
# CLIP score computation
# ---------------------------------------------------------------------------

def _to_tensor(out) -> torch.Tensor:
    """Handle both tensor and object returns from CLIP feature extractors."""
    if isinstance(out, torch.Tensor):
        return out
    # transformers ≥5.x wraps in a dataclass; the projected embedding is .image_embeds
    # or .text_embeds depending on the call, or fall back to pooler_output.
    for attr in ('image_embeds', 'text_embeds', 'pooler_output', 'last_hidden_state'):
        if hasattr(out, attr):
            val = getattr(out, attr)
            if val is not None:
                return val[:, 0] if val.ndim == 3 else val
    raise ValueError(f'Cannot extract tensor from CLIP output: {type(out)}')


def encode_texts(model, processor, texts: list[str], device: str) -> torch.Tensor:
    """Return L2-normalised text embeddings, shape (N, D)."""
    inputs = processor(
        text=texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=MAX_TEXT_LENGTH,
    ).to(device)
    with torch.no_grad():
        feats = _to_tensor(model.get_text_features(**inputs))
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu()


def encode_images(model, processor, images: list[Image.Image], device: str) -> torch.Tensor:
    """Return L2-normalised image embeddings, shape (N, D)."""
    inputs = processor(images=images, return_tensors='pt').to(device)
    with torch.no_grad():
        feats = _to_tensor(model.get_image_features(**inputs))
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu()


def compute_scores(records: list[dict], device: str) -> list[dict]:
    """Compute clip_gen, clip_gt, bleu1 for every record.

    Processes images and texts in batches for efficiency.
    clip_gt is the *maximum* CLIP similarity over all reference captions.
    """
    print(f'Loading CLIP model: {CLIP_MODEL_NAME}', flush=True)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model     = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    model.eval()

    n       = len(records)
    results = []

    for start in range(0, n, BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]

        # ── Load images ──────────────────────────────────────────────────
        images = []
        for rec in batch:
            img_path = os.path.join(VAL_IMAGE_DIR, rec['image'])
            images.append(Image.open(img_path).convert('RGB'))

        img_feats  = encode_images(model, processor, images, device)   # (B, D)

        # ── Encode generated captions ────────────────────────────────────
        gen_texts  = [rec['prediction'] for rec in batch]
        gen_feats  = encode_texts(model, processor, gen_texts, device)  # (B, D)
        clip_gen   = (img_feats * gen_feats).sum(dim=-1).tolist()       # (B,)

        # ── Encode references (all refs for the batch at once) ───────────
        # Flatten, encode, then reduce back per sample.
        ref_counts = [len(rec['references']) for rec in batch]
        flat_refs  = [r for rec in batch for r in rec['references'] if r.strip()]

        if flat_refs:
            ref_feats_flat = encode_texts(model, processor, flat_refs, device)
        else:
            ref_feats_flat = torch.zeros(0, img_feats.shape[-1])

        clip_gt = []
        ref_cursor = 0
        for i, rec in enumerate(batch):
            n_refs = ref_counts[i]
            if n_refs == 0:
                clip_gt.append(0.0)
                continue
            ref_feats = ref_feats_flat[ref_cursor:ref_cursor + n_refs]  # (n_refs, D)
            ref_cursor += n_refs
            # cosine sim of image i with each reference, take max
            sims = (img_feats[i].unsqueeze(0) * ref_feats).sum(dim=-1)  # (n_refs,)
            clip_gt.append(sims.max().item())

        # ── BLEU-1 ──────────────────────────────────────────────────────
        for i, rec in enumerate(batch):
            score = bleu1(rec['prediction'], rec['references'])
            results.append({
                'image':      rec['image'],
                'prediction': rec['prediction'],
                'references': rec['references'],
                'clip_gen':   clip_gen[i],
                'clip_gt':    clip_gt[i],
                'bleu1':      score,
            })

        if (start // BATCH_SIZE) % 5 == 0:
            print(f'  processed {min(start + BATCH_SIZE, n)}/{n}', flush=True)

    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_scatter_clip_vs_bleu(scores: list[dict], out_dir: str):
    """Scatter: CLIP(gen) on x, BLEU-1 on y. Colour = point density."""
    clip_vals = np.array([s['clip_gen'] for s in scores])
    bleu_vals = np.array([s['bleu1']    for s in scores]) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    h = ax.hist2d(clip_vals, bleu_vals, bins=60, cmap='YlOrRd')
    fig.colorbar(h[3], ax=ax, label='Sample count')

    ax.set_xlabel('CLIP score  (image ↔ generated caption)', fontsize=12)
    ax.set_ylabel('BLEU-1 (%)', fontsize=12)
    ax.set_title('CLIP Alignment vs BLEU-1\n'
                 'Bottom-right = semantically correct, BLEU-penalised', fontsize=12)

    # Quadrant lines at medians
    ax.axvline(np.median(clip_vals), color='steelblue', linestyle='--',
               linewidth=1, alpha=0.7, label=f'median CLIP={np.median(clip_vals):.3f}')
    ax.axhline(np.median(bleu_vals), color='seagreen', linestyle='--',
               linewidth=1, alpha=0.7, label=f'median BLEU={np.median(bleu_vals):.1f}%')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'scatter_clip_vs_bleu.png'), dpi=150)
    plt.close()
    print('[plot] scatter_clip_vs_bleu.png saved')


def plot_distributions(scores: list[dict], out_dir: str):
    """KDE / histogram overlay: distribution of clip_gen vs clip_gt."""
    clip_gen = np.array([s['clip_gen'] for s in scores])
    clip_gt  = np.array([s['clip_gt']  for s in scores])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overlaid histograms
    bins = np.linspace(
        min(clip_gen.min(), clip_gt.min()),
        max(clip_gen.max(), clip_gt.max()),
        60,
    )
    axes[0].hist(clip_gen, bins=bins, alpha=0.55, color='steelblue',
                 label=f'Generated  μ={clip_gen.mean():.3f}')
    axes[0].hist(clip_gt,  bins=bins, alpha=0.55, color='tomato',
                 label=f'Ground truth  μ={clip_gt.mean():.3f}')
    axes[0].set_xlabel('CLIP cosine similarity', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('CLIP Score Distribution', fontsize=12)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Right: clip_gen - clip_gt gap per sample
    gap = clip_gen - clip_gt
    axes[1].hist(gap, bins=60, color='mediumpurple', alpha=0.7)
    axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[1].axvline(gap.mean(), color='red', linestyle='-', linewidth=1.5,
                    label=f'mean gap={gap.mean():.3f}')
    axes[1].set_xlabel('CLIP(gen) − CLIP(gt)', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title('CLIP Gap: Generated vs Ground Truth\n'
                      '> 0 means generated caption aligns better with image', fontsize=11)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle('CLIP Alignment Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'distributions.png'), dpi=150)
    plt.close()
    print('[plot] distributions.png saved')


def plot_gen_vs_gt(scores: list[dict], out_dir: str):
    """Scatter: clip_gen vs clip_gt, coloured by BLEU-1."""
    clip_gen  = np.array([s['clip_gen'] for s in scores])
    clip_gt   = np.array([s['clip_gt']  for s in scores])
    bleu_vals = np.array([s['bleu1']    for s in scores])

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(clip_gt, clip_gen, c=bleu_vals * 100,
                    cmap='RdYlGn', s=4, alpha=0.5, vmin=0, vmax=60)
    fig.colorbar(sc, ax=ax, label='BLEU-1 (%)')

    lo = min(clip_gt.min(), clip_gen.min()) - 0.01
    hi = max(clip_gt.max(), clip_gen.max()) + 0.01
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, alpha=0.5, label='y = x')

    ax.set_xlabel('CLIP score  (image ↔ ground truth)', fontsize=11)
    ax.set_ylabel('CLIP score  (image ↔ generated)', fontsize=11)
    ax.set_title('Generated vs Ground-Truth CLIP Alignment\n'
                 'Colour = BLEU-1  |  Points above diagonal: gen > gt', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'scatter_gen_vs_gt.png'), dpi=150)
    plt.close()
    print('[plot] scatter_gen_vs_gt.png saved')


def plot_hidden_gems(scores: list[dict], out_dir: str, n: int = 8):
    """Gallery of samples with high CLIP(gen) but low BLEU-1.

    These are cases where the model is semantically on-target but gets
    penalised by n-gram overlap metrics.
    """
    clip_gen  = np.array([s['clip_gen'] for s in scores])
    bleu_vals = np.array([s['bleu1']    for s in scores])

    # High CLIP = top 30%; low BLEU = bottom 30%
    clip_thresh = np.percentile(clip_gen,  70)
    bleu_thresh = np.percentile(bleu_vals, 30)

    gems = [
        s for s in scores
        if s['clip_gen'] >= clip_thresh and s['bleu1'] <= bleu_thresh
    ]
    gems.sort(key=lambda s: s['clip_gen'] - s['bleu1'], reverse=True)
    gems = gems[:n]

    if not gems:
        print('[plot] No hidden gems found — skipping gallery.')
        return

    cols = min(4, len(gems))
    rows = (len(gems) + cols - 1) // cols
    fig  = plt.figure(figsize=(cols * 5, rows * 5.5))
    gs   = gridspec.GridSpec(rows, cols, figure=fig,
                             hspace=0.55, wspace=0.3)

    for idx, gem in enumerate(gems):
        r, c = divmod(idx, cols)
        ax   = fig.add_subplot(gs[r, c])

        img_path = os.path.join(VAL_IMAGE_DIR, gem['image'])
        if os.path.exists(img_path):
            ax.imshow(Image.open(img_path))
        else:
            ax.text(0.5, 0.5, '[image not found]',
                    ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

        gen_wrapped = '\n'.join(
            gem['prediction'][i:i + 38] for i in range(0, len(gem['prediction']), 38)
        )
        ref_short   = gem['references'][0][:60] + ('…' if len(gem['references'][0]) > 60 else '')

        ax.set_title(
            f"CLIP={gem['clip_gen']:.3f}  BLEU={gem['bleu1']*100:.1f}%\n"
            f"Gen: {gen_wrapped}\n"
            f"Ref: {ref_short}",
            fontsize=7,
            loc='left',
        )

    fig.suptitle(
        'Hidden Gems — High CLIP Alignment, Low BLEU Score\n'
        '(Model is semantically correct but penalised by n-gram mismatch)',
        fontsize=13,
    )
    plt.savefig(os.path.join(out_dir, 'hidden_gems.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print('[plot] hidden_gems.png saved')


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def print_summary(scores: list[dict]):
    clip_gen  = np.array([s['clip_gen'] for s in scores])
    clip_gt   = np.array([s['clip_gt']  for s in scores])
    bleu_vals = np.array([s['bleu1']    for s in scores])

    clip_thresh = np.percentile(clip_gen,  70)
    bleu_thresh = np.percentile(bleu_vals, 30)
    n_gems = sum(
        1 for s in scores
        if s['clip_gen'] >= clip_thresh and s['bleu1'] <= bleu_thresh
    )

    print('\n' + '=' * 60)
    print('CLIP ANALYSIS SUMMARY')
    print('=' * 60)
    print(f'  Samples analysed       : {len(scores):,}')
    print(f'  CLIP(gen)  mean ± std  : {clip_gen.mean():.4f} ± {clip_gen.std():.4f}')
    print(f'  CLIP(gt)   mean ± std  : {clip_gt.mean():.4f}  ± {clip_gt.std():.4f}')
    print(f'  CLIP gap   mean        : {(clip_gen - clip_gt).mean():+.4f}')
    print(f'  BLEU-1     mean        : {bleu_vals.mean()*100:.2f}%')
    print(f'  Corr(CLIP_gen, BLEU-1) : {np.corrcoef(clip_gen, bleu_vals)[0,1]:.4f}')
    print(f'  "Hidden gems"          : {n_gems:,}  '
          f'(top-30% CLIP ∩ bottom-30% BLEU)')
    pct_gen_beats_gt = (clip_gen > clip_gt).mean() * 100
    print(f'  Gen beats GT in CLIP   : {pct_gen_beats_gt:.1f}% of samples')
    print('=' * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='CLIP caption–image alignment analysis')
    p.add_argument('--predictions', default=DEFAULT_PREDS,
                   help='Path to predictions_val.json')
    p.add_argument('--use-cache', action='store_true',
                   help='Load scores.json from a previous run instead of recomputing')
    p.add_argument('--output-dir', default=_HERE,
                   help=f'Directory to save plots. Default: {_HERE}')
    p.add_argument('--gems-n', type=int, default=8,
                   help='Number of "hidden gem" samples to show in the gallery')
    return p.parse_args()


def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    os.makedirs(args.output_dir, exist_ok=True)

    cache_path = os.path.join(args.output_dir, 'scores.json')

    if args.use_cache and os.path.exists(cache_path):
        print(f'Loading cached scores from {cache_path}')
        with open(cache_path) as f:
            scores = json.load(f)
    else:
        print(f'Loading predictions from {args.predictions}')
        with open(args.predictions) as f:
            records = json.load(f)
        print(f'Computing CLIP scores for {len(records):,} samples ...')
        scores = compute_scores(records, device)
        with open(cache_path, 'w') as f:
            json.dump(scores, f)
        print(f'Scores cached to {cache_path}')

    print_summary(scores)

    print('\nGenerating plots ...')
    plot_scatter_clip_vs_bleu(scores, args.output_dir)
    plot_distributions(scores, args.output_dir)
    plot_gen_vs_gt(scores, args.output_dir)
    plot_hidden_gems(scores, args.output_dir, n=args.gems_n)

    print(f'\nAll outputs saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
