"""Training and evaluation loops for Task 2 (ViT + LoRA causal-LM captioning)."""
import os
import re
import time

import torch

from metrics import Metric

PRINT_EVERY = 20

# ---------------------------------------------------------------------------
# LoRA fine-tuning loops
# ---------------------------------------------------------------------------

def train_llama_one_epoch(
    model,
    optimizer,
    dataloader,
    scaler,
    device: str,
    epoch: int,
) -> float:
    """Single training epoch for ViTCausalLMCaptioningModel."""
    model.train()
    running_loss = 0.0
    t0 = time.time()

    for batch_idx, batch in enumerate(dataloader):
        pixel_values   = batch['pixel_values'].to(device, non_blocking=True)
        input_ids      = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels         = batch['labels'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
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
def eval_llama_epoch(
    model,
    metric: Metric,
    dataloader,
    tokenizer,
    device: str,
    epoch=None,
    generation_kwargs: dict | None = None,
) -> tuple[float, dict, list]:
    """Evaluate ViTCausalLMCaptioningModel; returns (avg_loss, metrics, predictions)."""
    model.eval()
    running_loss    = 0.0
    all_preds: list[str] = []
    all_refs:  list      = []
    predictions:   list  = []
    generation_kwargs = generation_kwargs or {}

    for batch_idx, batch in enumerate(dataloader):
        pixel_values   = batch['pixel_values'].to(device, non_blocking=True)
        input_ids      = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels         = batch['labels'].to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        running_loss += outputs.loss.item()

        pred_texts = model.generate(
            pixel_values=pixel_values,
            tokenizer=tokenizer,
            generation_kwargs=generation_kwargs,
        )

        all_preds.extend(pred_texts)
        all_refs.extend(batch['references'])

        for pred_text, img_name, refs in zip(
            pred_texts, batch['image_names'], batch['references']
        ):
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


def train_llama(
    model,
    optimizer,
    metric: Metric,
    dataloader_train,
    dataloader_val,
    tokenizer,
    epochs:    int,
    patience:  int,
    device:    str,
    save_path: str,
    min_delta: float = 1e-4,
    generation_kwargs: dict | None = None,
) -> tuple[list[dict], dict | None, list, int]:
    """Full training loop with early stopping for ViTCausalLMCaptioningModel.

    Returns (history, best_val_metrics, best_predictions, best_epoch).
    """
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    best_val_loss = float('inf')
    best_epoch    = -1
    best_metrics  = None
    best_preds:   list = []
    no_improve    = 0
    history:      list = []

    for epoch in range(epochs):
        t0 = time.time()

        train_loss = train_llama_one_epoch(
            model, optimizer, dataloader_train, scaler, device, epoch
        )
        print(f"[epoch {epoch}] train loss: {train_loss:.4f}", flush=True)

        val_loss, val_metrics, val_preds = eval_llama_epoch(
            model=model,
            metric=metric,
            dataloader=dataloader_val,
            tokenizer=tokenizer,
            device=device,
            epoch=epoch,
            generation_kwargs=generation_kwargs,
        )

        print(
            f"[epoch {epoch}] val loss: {val_loss:.4f} | "
            f"BLEU-1:{val_metrics['BLEU-1'] * 100:.1f}% "
            f"BLEU-2:{val_metrics['BLEU-2'] * 100:.1f}% "
            f"ROUGE-L:{val_metrics['ROUGE-L'] * 100:.1f}% "
            f"METEOR:{val_metrics['METEOR'] * 100:.1f}%",
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
            best_epoch    = epoch
            best_metrics  = val_metrics
            best_preds    = val_preds
            no_improve    = 0
            # Save directly to disk — avoids a deepcopy on GPU which OOMs on large models.
            torch.save(model.state_dict(), save_path)
            print(f"  → new best (val loss {best_val_loss:.4f}), saved to {save_path}", flush=True)
        else:
            no_improve += 1
            print(f"  → no improvement {no_improve}/{patience}", flush=True)
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location='cpu', weights_only=True))

    return history, best_metrics, best_preds, best_epoch


# ---------------------------------------------------------------------------
# Zero-shot evaluation with Qwen3-VL Vision-Language models
# ---------------------------------------------------------------------------

ZERO_SHOT_PROMPT = "Describe this image in one sentence."

# Pattern to strip Qwen3 <think>...</think> blocks if thinking mode fires.
_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)


def eval_qwen_zero_shot(
    metric:         Metric,
    dataloader,
    device:         str,
    model_name:     str = 'Qwen/Qwen3-VL-8B-Instruct',
    prompt:         str = ZERO_SHOT_PROMPT,
    max_new_tokens: int = 64,
) -> tuple[dict, list]:
    """Zero-shot evaluation with a Qwen3-VL Vision-Language model.

    Loads the model in bfloat16, runs zero-shot captioning on every image,
    then frees the model to avoid holding large allocations.

    Returns (metrics_dict, predictions_list).
    """
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    print(f"Loading zero-shot model: {model_name}", flush=True)
    processor = Qwen3VLProcessor.from_pretrained(model_name)
    processor.tokenizer.padding_side = 'left'
    zs_model  = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)
    zs_model.eval()

    # Chat template for a single image + text prompt (reused per image).
    messages_template = [
        {
            'role': 'user',
            'content': [
                {'type': 'image'},
                {'type': 'text', 'text': prompt},
            ],
        }
    ]
    input_text = processor.apply_chat_template(
        messages_template, tokenize=False, add_generation_prompt=True
    )

    all_preds:   list[str] = []
    all_refs:    list      = []
    predictions: list      = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images     = batch['images']       # list[PIL.Image] of length B
            img_names  = batch['image_names']
            references = batch['references']
            B          = len(images)

            # One processor call + one generate() for the whole batch.
            inputs = processor(
                text=[input_text] * B,
                images=images,
                return_tensors='pt',
                padding=True,
            ).to(device)

            generated_ids = zs_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

            # Strip the (padded) prompt tokens; same length for every item.
            input_len = inputs['input_ids'].shape[-1]
            for i, (img_name, refs) in enumerate(zip(img_names, references)):
                new_ids = generated_ids[i, input_len:]
                caption = processor.decode(new_ids, skip_special_tokens=True).strip()
                caption = _THINK_RE.sub('', caption).strip()

                all_preds.append(caption)
                all_refs.append(refs)
                predictions.append({
                    'image':      img_name,
                    'prediction': caption,
                    'references': list(refs),
                })

            if batch_idx % PRINT_EVERY == 0:
                print(f"  zero-shot batch {batch_idx}/{len(dataloader)}", flush=True)

    del zs_model
    if device == 'cuda':
        torch.cuda.empty_cache()

    metrics = metric(all_preds, all_refs)
    return metrics, predictions


# Backward-compatibility alias (old name used by older main.py versions).
eval_llama_zero_shot = eval_qwen_zero_shot
