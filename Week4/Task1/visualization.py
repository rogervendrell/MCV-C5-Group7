import csv
import json
import math
import os
import cv2
import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as _meteor_score
from rouge_score import rouge_scorer as _rouge_scorer

_HERE      = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_HERE, 'during_training_plots', 'pretrained')


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------

def save_gradcam_image(image_tensor, heatmap, token_text, epoch, output_dir=OUTPUT_DIR):
    """Overlay a GradCAM heatmap on a normalised image tensor and save it."""
    os.makedirs(output_dir, exist_ok=True)

    img_rgb = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean    = np.array([0.485, 0.456, 0.406])
    std     = np.array([0.229, 0.224, 0.225])
    img_rgb = np.clip(img_rgb * std + mean, 0, 1)

    heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap_color   = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0
    result          = np.clip(cv2.addWeighted(img_rgb, 0.6, heatmap_color, 0.4, 0), 0, 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(result)
    plt.title(f"Predicted Token: '{token_text}'", fontsize=16, pad=15)
    plt.axis('off')

    safe = token_text.replace('/', '_').replace('\\', '_')
    plt.savefig(os.path.join(output_dir, f"epoch_{epoch}_{safe}.png"),
                bbox_inches='tight', dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# ViT-GPT2 functions
# -----------------------------------------------------------------------------

def generate_word_gradcam(model, tokenizer, batch, device, epoch,
                          output_dir=OUTPUT_DIR):
    """Per-token GradCAM for a ViT-GPT2 VisionEncoderDecoderModel."""
    for param in model.parameters():
        param.requires_grad_(True)
    output_dir = os.path.join(output_dir, 'grad-cams')
    model.eval()
    torch.set_grad_enabled(True)

    pixel_values = batch['pixel_values'][0:1].to(device)
    pixel_values.requires_grad_(True)

    target_layer = model.model.encoder.encoder.layer[9].attention.output.dense
    activations  = []
    gradients    = []

    def _save_act(module, inp, output):
        activations.append(output)

    def _save_grad(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    h1 = target_layer.register_forward_hook(_save_act)
    h2 = target_layer.register_full_backward_hook(_save_grad)

    with torch.no_grad():
        gen_ids = model.generate(pixel_values, max_length=20)

    tokens = tokenizer.convert_ids_to_tokens(gen_ids[0])

    for i, token in enumerate(tokens):
        activations.clear()
        gradients.clear()

        clean_token = token.replace('Ġ', '').strip()
        if not clean_token or any(s in token for s in ['<|endoftext|>', '<pad>', '<s>']):
            continue

        model.zero_grad()
        outputs = model.model(
            pixel_values=pixel_values,
            decoder_input_ids=gen_ids,
            output_hidden_states=True,
        )
        score = outputs.logits[0, i, gen_ids[0, i]]
        score.backward(retain_graph=True)

        if not activations or not gradients:
            continue

        grad_tensor = gradients[0][0, 1:, :]   # drop CLS; (196, hidden)
        act_tensor  = activations[0][0, 1:, :]
        weights     = torch.mean(grad_tensor, dim=0)
        cam         = torch.relu((act_tensor * weights).sum(dim=-1)).reshape(14, 14)

        heatmap = cam.detach().cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        save_gradcam_image(pixel_values[0], heatmap, clean_token, epoch, output_dir)

    h1.remove()
    h2.remove()
    torch.set_grad_enabled(False)
    for param in model.parameters():
        param.requires_grad_(False)


def plot_token_importance(model, tokenizer, image_batch, device, epoch,
                          output_dir=OUTPUT_DIR):
    """Decoder self-attention heatmap for a ViT-GPT2 model."""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    gen_model     = model.model if hasattr(model, 'model') else model
    original_impl = getattr(gen_model.config, '_attn_implementation', 'sdpa')

    gen_model.config._attn_implementation = 'eager'
    gen_model.config.output_attentions    = True
    gen_model.config.output_hidden_states = False
    gen_model.config.add_cross_attention  = True

    try:
        pixel_values  = image_batch['pixel_values'][0:1].to(device)
        generated_ids = gen_model.generate(pixel_values, max_length=20)
        gen_ids       = generated_ids[0]
        all_tokens    = tokenizer.convert_ids_to_tokens(gen_ids)

        clean_indices = [i for i, t in enumerate(all_tokens)
                         if t not in tokenizer.all_special_tokens]
        clean_tokens  = [all_tokens[i].replace('Ġ', '') for i in clean_indices]

        if not clean_tokens:
            print('[viz] No valid tokens to plot.')
            return

        outputs = gen_model(
            pixel_values=pixel_values,
            labels=generated_ids,
            output_attentions=True,
            return_dict=True,
        )

        self_attn       = outputs.decoder_attentions[-1][0].mean(dim=0).detach().cpu().numpy()
        self_attn_clean = self_attn[np.ix_(clean_indices, clean_indices)]
        row_sums        = self_attn_clean.sum(axis=1, keepdims=True)
        self_attn_clean = np.divide(self_attn_clean, row_sums, where=row_sums != 0)

        plt.figure(figsize=(10, 8))
        sns.heatmap(self_attn_clean,
                    xticklabels=clean_tokens, yticklabels=clean_tokens,
                    annot=True, fmt='.2f', cmap='YlGnBu')
        plt.title(f'Self-Attention (Language) – Epoch {epoch}')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.savefig(os.path.join(output_dir, f'self_attn_epoch_{epoch}.png'),
                    bbox_inches='tight')
        plt.close()
        print(f'[viz] Self-attention plot saved for epoch {epoch}')

    except Exception as e:
        print(f'[viz] FAILED to plot token importance: {e}')

    finally:
        gen_model.config._attn_implementation = original_impl


# =============================================================================
# Per-epoch online visualizations (call from train loop like GradCAM)
# =============================================================================

def plot_cross_attention(model, tokenizer, image_batch, device, epoch,
                         output_dir=OUTPUT_DIR):
    """Cross-attention overlay """
    output_dir = os.path.join(output_dir, 'cross-attention')
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    gen_model     = model.model if hasattr(model, 'model') else model
    original_impl = getattr(gen_model.config, '_attn_implementation', 'sdpa')
    gen_model.config._attn_implementation = 'eager'

    try:
        pixel_values = image_batch['pixel_values'][0:1].to(device)

        with torch.no_grad():
            generated_ids = gen_model.generate(pixel_values, max_length=20)
            outputs = gen_model(
                pixel_values=pixel_values,
                labels=generated_ids,
                output_attentions=True,
                return_dict=True,
            )

        cross_attns = outputs.cross_attentions   # tuple: (batch, heads, dec_len, enc_len) per layer
        if not cross_attns:
            print('[viz] No cross_attentions returned — skipping.')
            return

        # Average over layers and heads → (dec_len, enc_len)
        avg_cross = torch.stack(
            [ca[0].mean(dim=0) for ca in cross_attns]
        ).mean(dim=0).detach().cpu().numpy()

        tokens      = tokenizer.convert_ids_to_tokens(generated_ids[0])
        clean_pairs = [
            (i, t.replace('Ġ', '').strip())
            for i, t in enumerate(tokens)
            if t not in tokenizer.all_special_tokens and t.replace('Ġ', '').strip()
        ]

        img_rgb = pixel_values[0].detach().cpu().numpy().transpose(1, 2, 0)
        mean    = np.array([0.485, 0.456, 0.406])
        std     = np.array([0.229, 0.224, 0.225])
        img_rgb = np.clip(img_rgb * std + mean, 0, 1)

        n_patches = 14   # ViT-B/16: 14×14 patches

        for token_idx, token_text in clean_pairs:
            if token_idx >= avg_cross.shape[0]:
                continue
            patch_attn = avg_cross[token_idx, 1:]   # drop CLS token
            if len(patch_attn) < n_patches * n_patches:
                continue

            heatmap = patch_attn[:n_patches * n_patches].reshape(n_patches, n_patches)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

            heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
            heatmap_color   = cv2.applyColorMap(
                np.uint8(255 * heatmap_resized), cv2.COLORMAP_VIRIDIS
            )
            heatmap_color   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0
            result          = np.clip(
                cv2.addWeighted(img_rgb, 0.55, heatmap_color, 0.45, 0), 0, 1
            )

            safe = token_text.replace('/', '_').replace('\\', '_')
            plt.figure(figsize=(6, 6))
            plt.imshow(result)
            plt.title(f"Cross-attn: '{token_text}' (epoch {epoch})", fontsize=14, pad=12)
            plt.axis('off')
            plt.savefig(
                os.path.join(output_dir, f'cross_attn_epoch{epoch}_{safe}.png'),
                bbox_inches='tight', dpi=150,
            )
            plt.close()

        print(f'[viz] Cross-attention maps saved for epoch {epoch}')

    except Exception as e:
        print(f'[viz] Cross-attention plot FAILED: {e}')
    finally:
        gen_model.config._attn_implementation = original_impl


def plot_token_log_probs(model, tokenizer, image_batch, device, epoch,
                         output_dir=OUTPUT_DIR):
    """Bar chart of the model's softmax probability for each generated token."""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    try:
        pixel_values = image_batch['pixel_values'][0:1].to(device)

        gen_model = model.model if hasattr(model, 'model') else model
        with torch.no_grad():
            generated_ids = gen_model.generate(pixel_values, max_length=20)
            outputs       = model(pixel_values=pixel_values, labels=generated_ids)

        logits = outputs.logits[0]                        # (seq_len, vocab)
        probs  = torch.softmax(logits, dim=-1).cpu()

        tokens      = tokenizer.convert_ids_to_tokens(generated_ids[0])
        clean_pairs = [
            (i, t.replace('Ġ', '').strip())
            for i, t in enumerate(tokens)
            if t not in tokenizer.all_special_tokens and t.replace('Ġ', '').strip()
        ]

        token_probs  = []
        token_labels = []
        for i, text in clean_pairs:
            if i < probs.shape[0] and i < generated_ids.shape[1]:
                token_probs.append(probs[i, generated_ids[0, i]].item())
                token_labels.append(text)

        if not token_probs:
            return

        fig_w = max(8, len(token_probs) * 0.7)
        plt.figure(figsize=(fig_w, 4))
        bars = plt.bar(range(len(token_probs)), token_probs,
                       color='steelblue', edgecolor='white')
        plt.xticks(range(len(token_labels)), token_labels,
                   rotation=40, ha='right', fontsize=10)
        plt.ylabel('Softmax Probability')
        plt.ylim(0, 1.05)
        plt.title(f'Token Generation Confidence — Epoch {epoch}', fontsize=13)
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='p = 0.5')
        plt.legend(fontsize=9)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f'token_probs_epoch_{epoch}.png'),
            dpi=150, bbox_inches='tight',
        )
        plt.close()
        print(f'[viz] Token probability plot saved for epoch {epoch}')

    except Exception as e:
        print(f'[viz] Token log-prob plot FAILED: {e}')


# =============================================================================
# Training dynamics tracker (integrate into train.py)
# =============================================================================

class GradientNormTracker:
    """Record per-layer gradient norms each epoch and plot a heatmap."""

    def __init__(self, model):
        self.model   = model
        self._records = []   # list of (epoch, layer_name, grad_norm)

    def record(self, epoch: int):
        """Call after loss.backward() (and scaler.unscale_ when using AMP)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self._records.append((epoch, name, param.grad.norm().item()))

    def plot(self, output_dir=OUTPUT_DIR):
        """Save a heatmap of mean gradient norm per (layer, epoch)."""
        if not self._records:
            print('[viz] GradientNormTracker: nothing recorded.')
            return

        os.makedirs(output_dir, exist_ok=True)

        epochs = sorted(set(r[0] for r in self._records))
        layers = list(dict.fromkeys(r[1] for r in self._records))   # preserves order

        ep_idx  = {e: i for i, e in enumerate(epochs)}
        ly_idx  = {l: i for i, l in enumerate(layers)}
        matrix  = np.zeros((len(layers), len(epochs)))
        counts  = np.zeros_like(matrix)

        for epoch, layer, norm in self._records:
            matrix[ly_idx[layer], ep_idx[epoch]] += norm
            counts[ly_idx[layer],  ep_idx[epoch]] += 1

        matrix = np.divide(matrix, counts, where=counts > 0)

        short = [l.replace('model.', '').replace('.weight', '') for l in layers]

        fig_h = max(8, len(layers) * 0.25)
        plt.figure(figsize=(max(10, len(epochs) * 1.2), fig_h))
        sns.heatmap(
            matrix,
            xticklabels=epochs,
            yticklabels=short,
            cmap='magma',
            linewidths=0,
            cbar_kws={'label': 'Gradient Norm'},
        )
        plt.title('Per-Layer Gradient Norms Over Training', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Layer')
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, 'gradient_norms_heatmap.png'),
            dpi=150, bbox_inches='tight',
        )
        plt.close()
        print(f'[viz] Gradient norm heatmap saved to {output_dir}')


# =============================================================================
# Post-hoc comparison plots  (call after all experiments finish)
# =============================================================================

def _per_sample_scores(predictions_json_path: str) -> list[dict]:
    """Compute per-sample BLEU-1, BLEU-2, ROUGE-L, and METEOR from the predictions JSON."""
    smoother = SmoothingFunction().method1
    rouge    = _rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    with open(predictions_json_path) as f:
        records = json.load(f)

    scores = []
    for rec in records:
        pred = rec['prediction'].strip()
        refs = [r.strip() for r in rec['references'] if r.strip()] or ['']

        pred_tok = pred.split()
        refs_tok  = [r.split() for r in refs]

        bleu1  = sentence_bleu(refs_tok, pred_tok,
                               weights=(1, 0, 0, 0),
                               smoothing_function=smoother)
        bleu2  = sentence_bleu(refs_tok, pred_tok,
                               weights=(0.5, 0.5, 0, 0),
                               smoothing_function=smoother)
        rougel = max(rouge.score(pred, r)['rougeL'].fmeasure for r in refs)
        met    = _meteor_score(refs_tok, pred_tok)

        scores.append({
            'BLEU-1':     bleu1,
            'BLEU-2':     bleu2,
            'ROUGE-L':    rougel,
            'METEOR':     met,
            'prediction': pred,
            'references': refs,
            'image':      rec.get('image', ''),
        })
    return scores


def plot_learning_curves(histories: dict, output_dir=OUTPUT_DIR):
    """Plot train/val loss and validation metrics over epochs for each method."""
    os.makedirs(output_dir, exist_ok=True)

    data = {}
    for label, csv_path in histories.items():
        rows = []
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                rows.append({k: float(v) for k, v in row.items()})
        data[label] = rows

    # --- Loss curves --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label, rows in data.items():
        eps   = [r['epoch'] for r in rows]
        axes[0].plot(eps, [r['train_loss'] for r in rows], marker='o', label=label)
        axes[1].plot(eps, [r['val_loss']   for r in rows], marker='s', label=label)
    for ax, title in zip(axes, ['Training Loss', 'Validation Loss']):
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves_loss.png'), dpi=150)
    plt.close()

    # -- Metric curves --------------------------------------------------------------
    metric_keys = ['val_BLEU-1', 'val_BLEU-2', 'val_ROUGE-L', 'val_METEOR']
    fig, axes   = plt.subplots(2, 2, figsize=(14, 10))
    for ax, key in zip(axes.flat, metric_keys):
        for label, rows in data.items():
            eps    = [r['epoch'] for r in rows]
            values = [r[key] * 100 for r in rows]
            ax.plot(eps, values, marker='o', label=label)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f'{key.replace("val_", "")} (%)')
        ax.set_title(key.replace('val_', ''))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.suptitle('Validation Metrics per Epoch', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves_metrics.png'), dpi=150)
    plt.close()
    print(f'[viz] Learning curves saved to {output_dir}')


def plot_metric_distributions(predictions: dict, output_dir=OUTPUT_DIR):
    """Violin + box plots of per-sample BLEU-1, BLEU-2, ROUGE-L, METEOR per method."""
    os.makedirs(output_dir, exist_ok=True)

    metric_names = ['BLEU-1', 'BLEU-2', 'ROUGE-L', 'METEOR']
    fig, axes    = plt.subplots(1, len(metric_names),
                                figsize=(6 * len(metric_names), 6))

    for ax, metric in zip(axes, metric_names):
        per_method = []
        method_labels = []
        for label, path in predictions.items():
            print(f'label: {label}, path: {path}')
            scores = _per_sample_scores(path)
            per_method.append([s[metric] * 100 for s in scores])
            method_labels.append(label)

        positions = list(range(1, len(per_method) + 1))
        vparts = ax.violinplot(per_method, positions=positions,
                               showmedians=True, showextrema=True)
        for pc in vparts['bodies']:
            pc.set_alpha(0.55)

        ax.boxplot(per_method, positions=positions, widths=0.12,
                   patch_artist=False,
                   medianprops=dict(color='black', linewidth=2),
                   showfliers=False)

        ax.set_xticks(positions)
        ax.set_xticklabels(method_labels, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel(f'{metric} (%)')
        ax.set_title(metric)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Per-Sample Score Distributions per Method', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_distributions.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[viz] Metric distributions saved to {output_dir}')


def plot_best_worst_gallery(predictions_json: str, image_dir: str,
                            output_dir=OUTPUT_DIR, n: int = 5,
                            sort_metric: str = 'METEOR'):
    """Save a side-by-side gallery of the n best and n worst predictions."""
    os.makedirs(output_dir, exist_ok=True)

    scores = _per_sample_scores(predictions_json)
    scores.sort(key=lambda x: x[sort_metric])
    worst = scores[:n]
    best  = scores[-n:][::-1]

    for split_name, split_data in [('best', best), ('worst', worst)]:
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
        if n == 1:
            axes = [axes]
        for ax, rec in zip(axes, split_data):
            img_path = os.path.join(image_dir, rec['image'])
            if os.path.exists(img_path):
                ax.imshow(plt.imread(img_path))
            else:
                ax.text(0.5, 0.5, '[image\nnot found]',
                        ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            score_str = f"{sort_metric}={rec[sort_metric] * 100:.1f}%"
            pred      = rec['prediction']
            # wrap prediction text at ~28 chars
            wrapped = '\n'.join(pred[i:i + 28] for i in range(0, len(pred), 28))
            ax.set_title(f"{score_str}\n{wrapped}", fontsize=7)

        plt.suptitle(
            f'{n} {split_name.capitalize()} Predictions by {sort_metric}',
            fontsize=13,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f'gallery_{split_name}.png'),
            dpi=150, bbox_inches='tight',
        )
        plt.close()

    print(f'[viz] Best/worst gallery saved to {output_dir}')


def plot_radar_chart(summaries: dict, output_dir=OUTPUT_DIR):
    """Radar / spider chart comparing BLEU-1, BLEU-2, ROUGE-L, METEOR per method."""
    os.makedirs(output_dir, exist_ok=True)

    metric_keys   = ['val_BLEU-1', 'val_BLEU-2', 'val_ROUGE-L', 'val_METEOR']
    metric_labels = ['BLEU-1', 'BLEU-2', 'ROUGE-L', 'METEOR']
    n = len(metric_keys)

    angles = [2 * math.pi * i / n for i in range(n)]
    angles += angles[:1]   # close polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for label, path in summaries.items():
        with open(path) as f:
            s = json.load(f)
        values = [s.get(k, 0.0) * 100 for k in metric_keys]
        values += values[:1]
        ax.plot(angles, values, marker='o', linewidth=2, label=label)
        ax.fill(angles, values, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_title('Method Comparison — Radar Chart', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15), fontsize=9)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radar_chart.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[viz] Radar chart saved to {output_dir}')


def plot_caption_length_analysis(predictions: dict, output_dir=OUTPUT_DIR):
    """Histogram and scatter of generated vs reference caption lengths per method."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    all_ref_lens, all_gen_lens = [], []   # for diagonal computation
    scatter_data = {}

    for label, path in predictions.items():
        with open(path) as f:
            records = json.load(f)
        gen_lens = [len(r['prediction'].split()) for r in records]
        ref_lens = [
            float(np.mean([len(ref.split()) for ref in r['references'] if ref.strip()]))
            for r in records
        ]
        scatter_data[label] = (ref_lens, gen_lens)
        all_ref_lens.extend(ref_lens)
        all_gen_lens.extend(gen_lens)

        axes[0].hist(gen_lens, bins=25, alpha=0.5, label=label)
        axes[1].scatter(ref_lens, gen_lens, alpha=0.25, s=8, label=label)

    axes[0].set_xlabel('Generated Caption Length (words)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Generated Caption Lengths')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # y = x diagonal
    lo = min(min(all_ref_lens), min(all_gen_lens))
    hi = max(max(all_ref_lens), max(all_gen_lens))
    axes[1].plot([lo, hi], [lo, hi], 'k--', alpha=0.5, linewidth=1.2, label='y = x')
    axes[1].set_xlabel('Mean Reference Length (words)')
    axes[1].set_ylabel('Generated Caption Length (words)')
    axes[1].set_title('Generated vs Reference Caption Length')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.suptitle('Caption Length Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'caption_length_analysis.png'), dpi=150)
    plt.close()
    print(f'[viz] Caption length analysis saved to {output_dir}')


def plot_vocabulary_analysis(predictions: dict, output_dir=OUTPUT_DIR):
    """Bar chart of vocabulary richness metrics per method."""
    os.makedirs(output_dir, exist_ok=True)

    stats = {}
    for label, path in predictions.items():
        with open(path) as f:
            records = json.load(f)
        all_tokens  = []
        cap_lengths = []
        for r in records:
            toks = r['prediction'].strip().split()
            all_tokens.extend(toks)
            cap_lengths.append(len(toks))
        n_total  = max(len(all_tokens), 1)
        n_unique = len(set(all_tokens))
        stats[label] = {
            'unique_tokens': n_unique,
            'ttr':           n_unique / n_total * 100,
            'avg_length':    float(np.mean(cap_lengths)),
        }

    labels  = list(stats.keys())
    x       = np.arange(len(labels))
    columns = [
        ('unique_tokens', 'Unique Token Types',        ''),
        ('ttr',           'Type-Token Ratio (TTR %)',   '%'),
        ('avg_length',    'Avg Caption Length (words)', ''),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (key, title, unit) in zip(axes, columns):
        values = [stats[l][key] for l in labels]
        bars   = ax.bar(x, values, width=0.5, color='steelblue', edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        y_pad = max(values) * 0.03
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_pad,
                f'{val:.1f}{unit}',
                ha='center', fontsize=9,
            )

    plt.suptitle('Vocabulary Richness per Method', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vocabulary_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[viz] Vocabulary analysis saved to {output_dir}')
