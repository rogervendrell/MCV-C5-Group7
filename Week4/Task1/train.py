"""Training and evaluation loops for ViT-GPT2 image captioning."""
import time
import copy

import torch
import torch.nn as nn

from metrics import Metric
from visualization import generate_word_gradcam, plot_token_importance

PRINT_EVERY = 20


def _compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy loss, optionally with label smoothing.

    VisionEncoderDecoderModel feeds labels directly (without shift) —
    the model handles the causal shift internally when building
    decoder_input_ids. The returned logits are aligned with the
    original labels, so we compute the loss without any extra shifting.
    Positions where labels == -100 are ignored.
    """
    vocab_size = logits.size(-1)
    loss_fct = nn.CrossEntropyLoss(
        label_smoothing=label_smoothing,
        ignore_index=-100,
    )
    return loss_fct(logits.view(-1, vocab_size), labels.view(-1))


def train_hf_one_epoch(
    model,
    optimizer,
    dataloader,
    scaler,
    device: str,
    epoch: int,
    label_smoothing: float = 0.0,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    running_loss = 0.0
    t0 = time.time()

    for batch_idx, batch in enumerate(dataloader):
        pixel_values = batch['pixel_values'].to(device, non_blocking=True)
        labels       = batch['labels'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
            outputs = model(pixel_values=pixel_values, labels=labels)
            if label_smoothing > 0.0:
                loss = _compute_loss(outputs.logits, labels, label_smoothing)
            else:
                loss = outputs.loss

        scaler.scale(loss).backward()
        if grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                max_norm=grad_clip,
            )
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        if batch_idx % PRINT_EVERY == 0:
            print(
                f"  epoch {epoch} | batch {batch_idx}/{len(dataloader)} "
                f"| loss {loss.item():.4f} | {time.time() - t0:.1f}s",
                flush=True,
            )

    return running_loss / len(dataloader)


@torch.no_grad()
def eval_hf_epoch(
    model,
    metric: Metric,
    dataloader,
    tokenizer,
    device: str,
    epoch=None,
    generation_kwargs: dict | None = None,
) -> tuple[float, dict, list]:
    """Returns (avg_loss, metrics_dict, predictions_list)."""
    model.eval()
    running_loss = 0.0
    all_preds, all_refs = [], []
    predictions = []
    generation_kwargs = generation_kwargs or {}

    for batch_idx, batch in enumerate(dataloader):
        pixel_values = batch['pixel_values'].to(device, non_blocking=True)
        labels       = batch['labels'].to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
            outputs  = model(pixel_values=pixel_values, labels=labels)
            loss_val = outputs.loss

        running_loss += loss_val.item()

        generated_ids = model.generate(pixel_values=pixel_values, **generation_kwargs)
        pred_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        pred_texts = [t.strip() for t in pred_texts]

        all_preds.extend(pred_texts)
        all_refs.extend(batch['references'])

        for pred_text, img_name, refs in zip(pred_texts, batch['image_names'], batch['references']):
            predictions.append({
                'image':      img_name,
                'prediction': pred_text,
                'references': list(refs),
            })

        if batch_idx % PRINT_EVERY == 0:
            ep = f"epoch {epoch} " if epoch is not None else ""
            print(f"  {ep}val batch {batch_idx}/{len(dataloader)}", flush=True)

    avg_loss = running_loss / len(dataloader)
    metrics  = metric(all_preds, all_refs)
    return avg_loss, metrics, predictions


def train_hf(
    model,
    optimizer,
    metric: Metric,
    dataloader_train,
    dataloader_val,
    tokenizer,
    epochs: int,
    patience: int,
    device: str,
    save_path: str,
    min_delta: float = 1e-4,
    label_smoothing: float = 0.0,
    grad_clip: float = 1.0,
    generation_kwargs: dict | None = None,
    viz_batch: dict | None = None,       # pass a batch here to enable per-epoch visualizations
    scheduler=None,
):
    """Full training loop with early stopping for HF captioning models.

    Visualization is supported: pass a `viz_batch` dict to enable per-epoch
    GradCAM and attention plots. When viz_batch is None (the default) the
    visualization code is skipped entirely — the functions remain importable
    and ready to use without any code changes.
    """
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    best_val_loss = float('inf')
    best_state    = None
    best_epoch    = -1
    best_metrics  = None
    best_preds    = []
    no_improve    = 0
    history       = []

    for epoch in range(epochs):
        t0 = time.time()

        train_loss = train_hf_one_epoch(
            model, optimizer, dataloader_train, scaler, device, epoch,
            label_smoothing=label_smoothing,
            grad_clip=grad_clip,
        )
        print(f"[epoch {epoch}] train loss: {train_loss:.4f}", flush=True)

        # --- Per-epoch visualizations (only when viz_batch is provided) ---
        if viz_batch is not None:
            try:
                generate_word_gradcam(model, tokenizer, viz_batch, device, epoch)
                plot_token_importance(model, tokenizer, viz_batch, device, epoch)
            except Exception as e:
                print(f"  [viz] skipped epoch {epoch}: {e}", flush=True)

        val_loss, val_metrics, val_preds = eval_hf_epoch(
            model=model,
            metric=metric,
            dataloader=dataloader_val,
            tokenizer=tokenizer,
            device=device,
            epoch=epoch,
            generation_kwargs=generation_kwargs,
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        print(
            f"[epoch {epoch}] val loss: {val_loss:.4f} | "
            f"BLEU-1:{val_metrics['BLEU-1']*100:.1f}% "
            f"BLEU-2:{val_metrics['BLEU-2']*100:.1f}% "
            f"ROUGE-L:{val_metrics['ROUGE-L']*100:.1f}% "
            f"METEOR:{val_metrics['METEOR']*100:.1f}% "
            f"sacreBLEU:{val_metrics['sacreBLEU']*100:.1f}%",
            flush=True,
        )

        history.append({
            'epoch':           epoch,
            'train_loss':      train_loss,
            'val_loss':        val_loss,
            'val_BLEU-1':      val_metrics['BLEU-1'],
            'val_BLEU-2':      val_metrics['BLEU-2'],
            'val_ROUGE-L':     val_metrics['ROUGE-L'],
            'val_METEOR':      val_metrics['METEOR'],
            'val_sacreBLEU':   val_metrics['sacreBLEU'],
            'epoch_time_sec':  time.time() - t0,
        })

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state    = copy.deepcopy(model.state_dict())
            best_epoch    = epoch
            best_metrics  = val_metrics
            best_preds    = val_preds
            no_improve    = 0
            torch.save(best_state, save_path)
            print(f"  → new best (val loss {best_val_loss:.4f}), saved to {save_path}", flush=True)
        else:
            no_improve += 1
            print(f"  → no improvement {no_improve}/{patience}", flush=True)
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history, best_metrics, best_preds, best_epoch
