"""Task 2: Image captioning with Qwen3 Vision-Language models.

Strategies
----------
zero_shot_8b
    Zero-shot evaluation with Qwen3-VL-8B-Instruct (comparable to Llama 3.2-11B).
    No training; the model is used out-of-the-box with a text prompt.

baseline_2b
    Zero-shot baseline with Qwen3-VL-2B-Instruct before LoRA fine-tuning.
    Run this before lora_2b to measure the model's native VL capability.

baseline_4b
    Zero-shot baseline with Qwen3-VL-4B-Instruct before LoRA fine-tuning.
    Run this before lora_4b to measure the model's native VL capability.

lora_2b
    Frozen ViT encoder  +  LoRA-fine-tuned Qwen3-1.7B decoder
    (comparable to Llama 3.2-1B).

lora_4b
    Frozen ViT encoder  +  LoRA-fine-tuned Qwen3-4B decoder
    (comparable to Llama 3.2-3B).

Architecture for LoRA strategies
---------------------------------
Frozen ViT (Task1_ft best weights) → linear projection → LoRA Qwen3 (text).
The visual backbone is the finetuned ViT from Task1_ft; only the projection
layer and the LoRA adapters are updated during training.
"""

import argparse
import csv
import glob
import json
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer

from dataset import (
    VizWizVisionDataset,
    LlamaLoRACollator,
    LlamaZeroShotCollator,
    load_vizwiz_split,
    get_dataset_paths,
)
from metrics import Metric
from model import ViTCausalLMCaptioningModel
from train import train_llama, eval_llama_epoch, eval_qwen_zero_shot

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Visual encoder: the Task1 model; its ViT encoder is extracted automatically.
VIT_MODEL_NAME = 'nlpconnect/vit-gpt2-image-captioning'

# Zero-shot VL models (Qwen3-VL, multimodal, used as-is)
QWEN_8B_MODEL_NAME    = 'Qwen/Qwen3-VL-8B-Instruct'   # comparable to Llama 11B
QWEN_VL_2B_MODEL_NAME = 'Qwen/Qwen3-VL-2B-Instruct'   # baseline before lora_2b
QWEN_VL_4B_MODEL_NAME = 'Qwen/Qwen3-VL-4B-Instruct'   # baseline before lora_4b

# Text-only Qwen3 decoders for LoRA fine-tuning with our custom ViT
QWEN_1_7B_MODEL_NAME = 'Qwen/Qwen3-1.7B'   # comparable to Llama 3.2-1B
QWEN_4B_MODEL_NAME   = 'Qwen/Qwen3-4B'     # comparable to Llama 3.2-3B

# Training hyper-parameters
EPOCHS            = 10
BATCH_SIZE        = 8
NUM_WORKERS       = 4
LR                = 2e-4
PATIENCE          = 5
MIN_DELTA         = 1e-4
MAX_TARGET_LENGTH = 64
NUM_BEAMS         = 4
MAX_NEW_TOKENS    = 64

# LoRA hyper-parameters
LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.1

DATASET_ROOT = None   # override via --dataset-root or VIZWIZ_ROOT env var

# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGIES = {
    'zero_shot_8b': {
        'label':          'Qwen3-VL-8B (zero-shot)',
        'eval_only':      True,
        'lm_model':       QWEN_8B_MODEL_NAME,
        'zs_batch_size':  1,   # ~19 GB used; only ~5 GB free on 24 GB GPU
    },
    'baseline_2b': {
        'label':          'Qwen3-VL-2B (zero-shot baseline)',
        'eval_only':      True,
        'lm_model':       QWEN_VL_2B_MODEL_NAME,
        'zs_batch_size':  8,
    },
    'baseline_4b': {
        'label':          'Qwen3-VL-4B (zero-shot baseline)',
        'eval_only':      True,
        'lm_model':       QWEN_VL_4B_MODEL_NAME,
        'zs_batch_size':  4,
    },
    'lora_2b': {
        'label':            'ViT frozen | Qwen3-1.7B LoRA',
        'eval_only':        False,
        'lm_model':         QWEN_1_7B_MODEL_NAME,
        'train_batch_size': 4,
        'val_batch_size':   4,
    },
    'lora_4b': {
        'label':            'ViT frozen | Qwen3-4B LoRA',
        'eval_only':        False,
        'lm_model':         QWEN_4B_MODEL_NAME,
        'train_batch_size': 8,
        'val_batch_size':   8,
    },
}

_HERE        = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(_HERE, 'results')
METRIC_COLUMNS = ['BLEU-1', 'BLEU-2', 'ROUGE-L', 'METEOR']

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_results_dir(base_root: str, experiment_name: str) -> tuple[str, str, str]:
    job_id       = os.environ.get('SLURM_JOB_ID', 'local')
    results_dir  = os.path.join(base_root, experiment_name, job_id)
    weights_dir  = os.path.join(results_dir, 'weights')
    best_weights = os.path.join(weights_dir, 'best_model.pt')
    os.makedirs(weights_dir, exist_ok=True)
    return results_dir, weights_dir, best_weights


def save_history_csv(history: list[dict], path: str):
    if not history:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def write_experiment_report(results_root: str):
    summary_paths = sorted(
        glob.glob(os.path.join(results_root, '**', 'summary.json'), recursive=True)
    )
    summaries = []
    for path in summary_paths:
        with open(path) as f:
            s = json.load(f)
        if s.get('family') != 'vit-qwen':
            continue
        summaries.append(s)

    if not summaries:
        return

    report_rows = [
        {
            'experiment': s.get('label', s.get('experiment', 'unknown')),
            'family':     s.get('family', ''),
            'BLEU-1':     s.get('val_BLEU-1'),
            'BLEU-2':     s.get('val_BLEU-2'),
            'ROUGE-L':    s.get('val_ROUGE-L'),
            'METEOR':     s.get('val_METEOR'),
        }
        for s in summaries
    ]

    report_csv = os.path.join(results_root, 'experiment_report.csv')
    report_md  = os.path.join(results_root, 'experiment_report.md')

    with open(report_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['experiment', 'family', *METRIC_COLUMNS])
        writer.writeheader()
        writer.writerows(report_rows)

    with open(report_md, 'w') as f:
        headers = ['Experiment', 'Family', *METRIC_COLUMNS]
        f.write('| ' + ' | '.join(headers) + ' |\n')
        f.write('| ' + ' | '.join(['---'] * len(headers)) + ' |\n')
        for row in report_rows:
            values = []
            for col in METRIC_COLUMNS:
                v = row[col]
                values.append(f"{v * 100:.2f}" if v is not None else 'NA')
            f.write(f"| {row['experiment']} | {row['family']} | " + ' | '.join(values) + ' |\n')


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Task 2: ViT + Qwen3 image captioning')
    parser.add_argument(
        '--strategy', choices=list(STRATEGIES), default='lora_2b',
        help='Which experiment to run',
    )
    parser.add_argument(
        '--vit-model', default=None,
        help='Override ViT model name (default: VIT_MODEL_NAME constant)',
    )
    parser.add_argument(
        '--vit-checkpoint', default=None,
        help='Path to a Task1_ft best_model.pt to initialise the ViT encoder',
    )
    parser.add_argument(
        '--dataset-root', default=None,
        help='Root directory of the VizWiz dataset (overrides VIZWIZ_ROOT env var)',
    )
    parser.add_argument(
        '--report-only', action='store_true',
        help='Regenerate the experiment report without running any experiment',
    )
    parser.add_argument(
        '--eval-only', action='store_true',
        help='For LoRA strategies: skip training, load best_model.pt and run one val pass',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    if args.report_only:
        write_experiment_report(RESULTS_ROOT)
        print(f"Report updated in: {RESULTS_ROOT}")
        return

    dataset_root   = args.dataset_root or DATASET_ROOT
    dataset_paths  = get_dataset_paths(dataset_root)
    train_json     = dataset_paths['train_json']
    val_json       = dataset_paths['val_json']
    train_img_path = dataset_paths['train_img_path']
    val_img_path   = dataset_paths['val_img_path']

    print(f"Device:   {DEVICE}", flush=True)
    print(f"Strategy: {args.strategy}", flush=True)

    train_samples = load_vizwiz_split(train_json)
    val_samples   = load_vizwiz_split(val_json)
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}", flush=True)

    metric          = Metric()
    strategy_cfg    = STRATEGIES[args.strategy]
    experiment_name = f"qwen_{args.strategy}"
    results_dir, _, best_weights = build_results_dir(RESULTS_ROOT, experiment_name)

    vit_model_name = args.vit_model or VIT_MODEL_NAME
    lm_model_name  = strategy_cfg['lm_model']

    # ------------------------------------------------------------------
    # Zero-shot / baseline evaluation (Qwen3-VL multimodal models)
    # ------------------------------------------------------------------
    if strategy_cfg['eval_only']:
        dl_val_zs = DataLoader(
            VizWizVisionDataset(val_samples, val_img_path),
            batch_size=strategy_cfg['zs_batch_size'],
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=LlamaZeroShotCollator(),
        )

        best_metrics, best_preds = eval_qwen_zero_shot(
            metric=metric,
            dataloader=dl_val_zs,
            device=DEVICE,
            model_name=lm_model_name,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        best_epoch = -1
        history    = []
        n_params   = 0

    # ------------------------------------------------------------------
    # LoRA fine-tuning (Qwen3-1.7B or Qwen3-4B text decoder + frozen ViT)
    # ------------------------------------------------------------------
    else:
        # Image processor comes from the ViT model
        image_processor = AutoImageProcessor.from_pretrained(vit_model_name)

        # Qwen3 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token    = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'right'

        collator = LlamaLoRACollator(
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_target_length=MAX_TARGET_LENGTH,
        )

        dl_train = DataLoader(
            VizWizVisionDataset(train_samples, train_img_path),
            batch_size=strategy_cfg['train_batch_size'],
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=(DEVICE == 'cuda'),
            persistent_workers=(NUM_WORKERS > 0),
            collate_fn=collator,
        )
        dl_val = DataLoader(
            VizWizVisionDataset(val_samples, val_img_path),
            batch_size=strategy_cfg['val_batch_size'],
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(DEVICE == 'cuda'),
            persistent_workers=(NUM_WORKERS > 0),
            collate_fn=collator,
        )

        # Build model: frozen ViT → projection → LoRA Qwen3
        model = ViTCausalLMCaptioningModel(
            vit_model_name=vit_model_name,
            lm_model_name=lm_model_name,
            lora_r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
        ).to(DEVICE)

        # Optionally load fine-tuned ViT encoder weights from Task1_ft
        if args.vit_checkpoint:
            model.encoder.load_task1_encoder_weights(args.vit_checkpoint)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {n_params:,}", flush=True)

        # Load checkpoint (required for --eval-only, optional resume otherwise)
        if os.path.exists(best_weights):
            print(f"Loading weights from: {best_weights}", flush=True)
            state = torch.load(best_weights, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state)
        elif args.eval_only:
            raise FileNotFoundError(f"--eval-only requires a checkpoint at {best_weights}")

        generation_kwargs = {
            'max_new_tokens': MAX_NEW_TOKENS,
            'num_beams':      NUM_BEAMS,
            'eos_token_id':   tokenizer.eos_token_id,
            'pad_token_id':   tokenizer.pad_token_id,
        }

        if args.eval_only:
            print("Eval-only mode: running single validation pass.", flush=True)
            val_loss, best_metrics, best_preds = eval_llama_epoch(
                model=model, metric=metric, dataloader=dl_val,
                tokenizer=tokenizer, device=DEVICE, epoch=None,
                generation_kwargs=generation_kwargs,
            )
            history, best_epoch = [], -1
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=LR,
            )

        if not args.eval_only:
            history, best_metrics, best_preds, best_epoch = train_llama(
                model=model,
                optimizer=optimizer,
                metric=metric,
                dataloader_train=dl_train,
                dataloader_val=dl_val,
                tokenizer=tokenizer,
                epochs=EPOCHS,
                patience=PATIENCE,
                device=DEVICE,
                save_path=best_weights,
                min_delta=MIN_DELTA,
                generation_kwargs=generation_kwargs,
            )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    summary = {
        'family':               'vit-qwen',
        'experiment':           experiment_name,
        'label':                strategy_cfg['label'],
        'vit_model_name':       vit_model_name,
        'lm_model_name':        lm_model_name,
        'best_epoch':           best_epoch,
        'num_trainable_params': n_params,
        'lora_r':               LORA_R,
        'lora_alpha':           LORA_ALPHA,
        'lora_dropout':         LORA_DROPOUT,
        **{f'val_{k}': v for k, v in (best_metrics or {}).items()},
    }

    save_history_csv(history, os.path.join(results_dir, 'history.csv'))

    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(results_dir, 'predictions_val.json'), 'w') as f:
        json.dump(best_preds, f, indent=2)

    write_experiment_report(RESULTS_ROOT)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('FINAL RESULTS')
    print('=' * 60)
    print(f"Best epoch : {best_epoch}")
    if best_metrics:
        for k, v in best_metrics.items():
            print(f"  {k}: {v * 100:.2f}%")
    print(f"\nResults saved to: {results_dir}")
    if history:
        print(f"Best weights   : {best_weights}")
    print(f"Report table   : {os.path.join(RESULTS_ROOT, 'experiment_report.md')}")


if __name__ == '__main__':
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    main()
