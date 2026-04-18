"""Task E: ViT + LoRA Qwen3 captioning with augmented training data.

Same architecture as Week4/Task2 lora_2b:
    Frozen ViT (Task1_ft best weights) → linear projection → LoRA Qwen3-1.7B

New: the training set can be augmented with degraded images generated in Task D.
The fraction of augmented data is controlled via --aug-fraction (0–100).
"""

import argparse
import csv
import glob
import json
import os

import torch
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoImageProcessor, AutoTokenizer

from dataset import (
    VizWizVisionDataset,
    LlamaLoRACollator,
    load_vizwiz_split,
    get_dataset_paths,
    load_augmented_samples,
)
from metrics import Metric
from model import ViTCausalLMCaptioningModel
from train import train_llama, eval_llama_epoch, load_resume_checkpoint

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VIT_MODEL_NAME   = 'nlpconnect/vit-gpt2-image-captioning'
QWEN_1_7B_MODEL_NAME = 'Qwen/Qwen3-1.7B'

EPOCHS            = 10
BATCH_SIZE        = 4
NUM_WORKERS       = 4
LR                = 2e-4
PATIENCE          = 5
MIN_DELTA         = 1e-4
MAX_TARGET_LENGTH = 64
NUM_BEAMS         = 4
MAX_NEW_TOKENS    = 64

LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.1

# Default augmented dataset produced by Task D
DEFAULT_AUG_DIR = '/ghome/group07/MCV-C5-Group7/Week5/task_d/augmented_dataset/110762'

_HERE        = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(_HERE, 'results')
METRIC_COLUMNS = ['BLEU-1', 'BLEU-2', 'ROUGE-L', 'METEOR']

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_results_dir(base_root: str, experiment_name: str) -> tuple[str, str, str]:
    job_id      = os.environ.get('SLURM_JOB_ID', 'local')
    results_dir = os.path.join(base_root, experiment_name, job_id)
    weights_dir = os.path.join(results_dir, 'weights')
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
            'aug_fraction': s.get('aug_fraction', 0),
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
        writer = csv.DictWriter(f, fieldnames=['experiment', 'family', 'aug_fraction', *METRIC_COLUMNS])
        writer.writeheader()
        writer.writerows(report_rows)

    with open(report_md, 'w') as f:
        headers = ['Experiment', 'Family', 'Aug%', *METRIC_COLUMNS]
        f.write('| ' + ' | '.join(headers) + ' |\n')
        f.write('| ' + ' | '.join(['---'] * len(headers)) + ' |\n')
        for row in report_rows:
            values = []
            for col in METRIC_COLUMNS:
                v = row[col]
                values.append(f"{v * 100:.2f}" if v is not None else 'NA')
            aug_pct = row.get('aug_fraction', 0)
            f.write(f"| {row['experiment']} | {row['family']} | {aug_pct} | " + ' | '.join(values) + ' |\n')


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Task E: ViT + Qwen3-1.7B LoRA with augmented data')
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
        '--aug-fraction', type=int, default=0,
        metavar='PCT',
        help='Percentage of augmented (Task D) data to include in training [0–100] (default: 0)',
    )
    parser.add_argument(
        '--aug-dataset', default=DEFAULT_AUG_DIR,
        help='Path to the Task D augmented dataset directory (must contain annotations.json and images/)',
    )
    parser.add_argument(
        '--eval-only', action='store_true',
        help='Skip training, load best_model.pt and run one val pass',
    )
    parser.add_argument(
        '--resume-dir', default=None, metavar='PATH',
        help='Path to a previous run\'s results directory (the one containing '
             'weights/resume.pt).  Training resumes from that checkpoint and '
             'new outputs are written to the same directory.',
    )
    parser.add_argument(
        '--report-only', action='store_true',
        help='Regenerate the experiment report without running any experiment',
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

    aug_fraction = max(0, min(100, args.aug_fraction))
    aug_label    = f"Aug{aug_fraction}pct"
    label        = f"ViT frozen | Qwen3-1.7B LoRA + {aug_label}"
    experiment_name = f"qwen_lora_2b_{aug_label}"

    dataset_root   = args.dataset_root
    dataset_paths  = get_dataset_paths(dataset_root)
    train_img_path = dataset_paths['train_img_path']
    val_img_path   = dataset_paths['val_img_path']

    print(f"Device:        {DEVICE}", flush=True)
    print(f"Experiment:    {experiment_name}", flush=True)
    print(f"Aug fraction:  {aug_fraction}%", flush=True)

    # Load original VizWiz splits
    train_samples = load_vizwiz_split(dataset_paths['train_json'])
    val_samples   = load_vizwiz_split(dataset_paths['val_json'])
    print(f"Original train: {len(train_samples)},  Val: {len(val_samples)}", flush=True)

    # Build training dataset
    orig_train_ds = VizWizVisionDataset(train_samples, train_img_path)

    if aug_fraction > 0:
        aug_json     = os.path.join(args.aug_dataset, 'annotations.json')
        aug_img_path = os.path.join(args.aug_dataset, 'images')
        aug_samples  = load_augmented_samples(aug_json, fraction=aug_fraction / 100.0)
        aug_train_ds = VizWizVisionDataset(aug_samples, aug_img_path)
        train_ds     = ConcatDataset([orig_train_ds, aug_train_ds])
        print(f"Augmented samples added: {len(aug_samples)}", flush=True)
    else:
        train_ds = orig_train_ds

    print(f"Total training samples: {len(train_ds)}", flush=True)

    val_ds = VizWizVisionDataset(val_samples, val_img_path)

    if args.resume_dir:
        # Reuse the exact directory from the previous run so weights are found.
        results_dir  = args.resume_dir
        weights_dir  = os.path.join(results_dir, 'weights')
        best_weights = os.path.join(weights_dir, 'best_model.pt')
        os.makedirs(weights_dir, exist_ok=True)
    else:
        results_dir, weights_dir, best_weights = build_results_dir(RESULTS_ROOT, experiment_name)

    vit_model_name = args.vit_model or VIT_MODEL_NAME
    lm_model_name  = QWEN_1_7B_MODEL_NAME

    image_processor = AutoImageProcessor.from_pretrained(vit_model_name)

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
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == 'cuda'),
        persistent_workers=(NUM_WORKERS > 0),
        collate_fn=collator,
    )
    dl_val = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == 'cuda'),
        persistent_workers=(NUM_WORKERS > 0),
        collate_fn=collator,
    )

    model = ViTCausalLMCaptioningModel(
        vit_model_name=vit_model_name,
        lm_model_name=lm_model_name,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
    ).to(DEVICE)

    if args.vit_checkpoint:
        model.encoder.load_task1_encoder_weights(args.vit_checkpoint)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}", flush=True)

    resume_path = os.path.join(weights_dir, 'resume.pt')

    if args.resume_dir:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"--resume-dir requires a checkpoint at {resume_path}")
        print(f"Will resume from: {resume_path}", flush=True)
        # resume.pt is loaded inside train_llama; skip the best_weights preload here.
    elif os.path.exists(best_weights):
        print(f"Loading weights from: {best_weights}", flush=True)
        state = torch.load(best_weights, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
    elif args.eval_only:
        raise FileNotFoundError(f"--eval-only requires a checkpoint at {best_weights}")

    metric = Metric()

    generation_kwargs = {
        'max_new_tokens': MAX_NEW_TOKENS,
        'num_beams':      NUM_BEAMS,
        'eos_token_id':   tokenizer.eos_token_id,
        'pad_token_id':   tokenizer.pad_token_id,
    }

    if args.eval_only:
        print("Eval-only mode: running single validation pass.", flush=True)
        _, best_metrics, best_preds = eval_llama_epoch(
            model=model, metric=metric, dataloader=dl_val,
            tokenizer=tokenizer, device=DEVICE, epoch=None,
            generation_kwargs=generation_kwargs,
        )
        history, best_epoch = [], -1
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=LR,
        )
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
            resume_checkpoint=resume_path if args.resume_dir else None,
        )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    summary = {
        'family':               'vit-qwen',
        'experiment':           experiment_name,
        'label':                label,
        'aug_fraction':         aug_fraction,
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
