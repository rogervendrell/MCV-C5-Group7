"""Training and evaluation loop utilities for the baseline model."""
import time
import copy

import torch
from torch import nn

import tokenizer as tok
from metrics import Metric, decode_caption

PRINT_EVERY = 20


def train_one_epoch(
    model,
    optimizer,
    criterion,
    dataloader,
    scaler,
    device: str,
    epoch: int,
) -> float:
    model.train()
    running_loss = 0.0
    t0 = time.time()

    for batch_idx, (img, caption, *_) in enumerate(dataloader):
        img     = img.to(device, non_blocking=True)
        caption = caption.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
            pred = model(img, ground_truth=caption)
            loss = criterion(pred, caption)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        if batch_idx % PRINT_EVERY == 0:
            print(
                f"  epoch {epoch} | batch {batch_idx}/{len(dataloader)} "
                f"| loss {loss.item():.4f} | {time.time()-t0:.1f}s",
                flush=True,
            )

    return running_loss / len(dataloader)


@torch.no_grad()
def eval_epoch(
    model,
    criterion,
    metric: Metric,
    dataloader,
    device: str,
    epoch=None,
) -> tuple[float, dict, list]:
    """Returns (avg_loss, metrics_dict, predictions_list).

    predictions_list entries: {'image': str, 'prediction': str, 'references': list[str]}
    """
    model.eval()
    running_loss = 0.0
    all_preds, all_refs = [], []
    predictions = []

    for batch_idx, (img, caption, img_names, all_captions) in enumerate(dataloader):
        img     = img.to(device, non_blocking=True)
        caption = caption.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
            pred      = model(img)
            loss_val  = criterion(pred, caption)

        running_loss += loss_val.item()

        pred_ids = pred.argmax(dim=1).cpu().numpy()
        cap_ids  = caption.cpu().numpy()

        all_preds.extend(pred_ids)
        all_refs.extend(cap_ids)

        # Collect per-image predictions for the eval framework
        for pred_seq, img_name, caps in zip(pred_ids, img_names, all_captions):
            predictions.append({
                'image':      img_name,
                'prediction': decode_caption(pred_seq),
                'references': list(caps),
            })

        if batch_idx % PRINT_EVERY == 0:
            ep = f"epoch {epoch} " if epoch is not None else ""
            print(f"  {ep}val batch {batch_idx}/{len(dataloader)}", flush=True)

    avg_loss = running_loss / len(dataloader)
    metrics  = metric(all_preds, all_refs)

    # --- SANITY CHECK PRINTS ---
    print("\n" + "="*50)
    print(f" EPOCH {epoch if epoch is not None else 'N/A'} - SAMPLE PREDICTIONS")
    print("="*50)

    return avg_loss, metrics, predictions


def train(
    model,
    optimizer,
    criterion,
    metric: Metric,
    dataloader_train,
    dataloader_val,
    epochs: int,
    patience: int,
    device: str,
    save_path: str,
    min_delta: float = 1e-4,
):
    """Full training loop with early stopping and best-model checkpointing.

    Returns (history: list[dict], best_val_metrics: dict, best_predictions: list).
    """
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    best_val_loss  = float('inf')
    best_state     = None
    best_epoch     = -1
    best_metrics   = None
    best_preds     = []
    no_improve     = 0
    history        = []

    for epoch in range(epochs):
        t0 = time.time()

        train_loss = train_one_epoch(model, optimizer, criterion, dataloader_train, scaler, device, epoch)
        print(f"[epoch {epoch}] train loss: {train_loss:.4f}", flush=True)

        val_loss, val_metrics, val_preds = eval_epoch(model, criterion, metric, dataloader_val, device, epoch)

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

        # Print 3 examples from the current validation batch
        import random
        indices = random.sample(range(len(val_preds)), min(3, len(val_preds)))
        
        for idx in indices:
            item = val_preds[idx]
            print(f"IMAGE: {item['image']}")
            print(f"PRED:  {item['prediction']}")
            print(f"REF:   {item['references'][0]}") # Show the first reference
            print("-" * 30)
        print("="*50 + "\n")

    if best_state is not None:
        model.load_state_dict(best_state)

    return history, best_metrics, best_preds, best_epoch
