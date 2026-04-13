"""ViT-GPT2 image captioning — regular fine-tuning experiments.

Uses the native GPT-2 BPE tokenizer throughout (model input, label encoding,
and metric evaluation). No custom vocabulary.

Fine-tuning strategies
----------------------
pretrained        – both components frozen, evaluation only
vit_ft_gpt_frozen – encoder fine-tuned (last N layers), decoder frozen
vit_frozen_gpt_ft – encoder frozen, decoder fine-tuned (cross-attention only
                    by default — preserves language generation, avoids
                    repetition collapse)
vit_ft_gpt_ft     – both components fine-tuned

Key regularization levers (all configurable via CLI)
-----------------------------------------------------
--lr              Learning rate (default 1e-5, lower than naive 5e-5)
--weight-decay    AdamW weight decay (default 0.01)
--label-smoothing Label smoothing coefficient (default 0.1)
--grad-clip       Gradient clipping max-norm (default 1.0)
--repetition-penalty  Generation penalty > 1.0 suppresses repeated phrases
                      (default 1.3)
--encoder-mode    frozen | last_n | full
--decoder-mode    frozen | cross_attn | last_n | full
--encoder-ft-layers   N blocks from top of ViT to unfreeze (last_n mode)
--decoder-ft-layers   N blocks from top of GPT-2 to unfreeze (last_n mode)
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
    VisionEncoderDecoderCollator,
    load_vizwiz_split,
    get_dataset_paths,
)
from metrics import Metric
from model import VisionEncoderDecoderCaptioningModel, ENCODER_MODES, DECODER_MODES
from train import train_hf, eval_hf_epoch

# ── Defaults ──────────────────────────────────────────────────────────────────
EPOCHS            = 10
BATCH_SIZE        = 16
NUM_WORKERS       = 4
LR                = 1e-5
WEIGHT_DECAY      = 0.01
LABEL_SMOOTHING   = 0.1
GRAD_CLIP         = 1.0
PATIENCE          = 10
MIN_DELTA         = 1e-4
HF_MODEL_NAME     = 'nlpconnect/vit-gpt2-image-captioning'
MAX_TARGET_LENGTH = 64
NUM_BEAMS         = 4
REPETITION_PENALTY = 1.3
ENCODER_FT_LAYERS = 4
DECODER_FT_LAYERS = 4
DATASET_ROOT      = None

_HERE        = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(_HERE, 'results')
METRIC_COLS  = ['BLEU-1', 'BLEU-2', 'ROUGE-L', 'METEOR']

# ── Strategy table ────────────────────────────────────────────────────────────
# encoder_mode / decoder_mode defaults; overridable via CLI.
STRATEGIES = {
    'pretrained': {
        'encoder_mode': 'frozen',
        'decoder_mode': 'frozen',
        'eval_only':    True,
        'label':        'ViT-GPT2 pretrained (frozen)',
    },
    'vit_ft_gpt_frozen': {
        'encoder_mode': 'last_n',
        'decoder_mode': 'frozen',
        'eval_only':    False,
        'label':        'ViT last-N FT | GPT2 frozen',
    },
    'vit_frozen_gpt_ft': {
        'encoder_mode': 'frozen',
        'decoder_mode': 'cross_attn',   # safest decoder FT: cross-attn only
        'eval_only':    False,
        'label':        'ViT frozen | GPT2 cross-attn FT',
    },
    'vit_ft_gpt_ft': {
        'encoder_mode': 'last_n',
        'decoder_mode': 'cross_attn',
        'eval_only':    False,
        'label':        'ViT last-N FT | GPT2 cross-attn FT',
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def make_dataloader(samples, img_path, shuffle, batch_size, num_workers,
                    image_processor, tokenizer, max_target_length, device):
    ds = VizWizVisionDataset(samples, img_path)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device == 'cuda'),
        persistent_workers=(num_workers > 0),
        collate_fn=VisionEncoderDecoderCollator(
            image_processor, tokenizer,
            max_target_length=max_target_length,
        ),
    )


def write_experiment_report(results_root: str):
    summary_paths = sorted(
        glob.glob(os.path.join(results_root, '**', 'summary.json'), recursive=True)
    )
    summaries = []
    for path in summary_paths:
        with open(path) as f:
            s = json.load(f)
        if s.get('family') != 'vit-gpt2':
            continue
        summaries.append(s)

    if not summaries:
        return

    report_csv = os.path.join(results_root, 'experiment_report.csv')
    report_md  = os.path.join(results_root, 'experiment_report.md')
    csv_cols   = ['experiment', 'family', *METRIC_COLS]

    rows = []
    for s in summaries:
        rows.append({
            'experiment': s.get('label', s.get('experiment', '?')),
            'family':     s.get('family', ''),
            'BLEU-1':     s.get('val_BLEU-1'),
            'BLEU-2':     s.get('val_BLEU-2'),
            'ROUGE-L':    s.get('val_ROUGE-L'),
            'METEOR':     s.get('val_METEOR'),
        })

    with open(report_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        writer.writerows(rows)

    with open(report_md, 'w') as f:
        headers = ['Experiment', 'Family', *METRIC_COLS]
        f.write('| ' + ' | '.join(headers) + ' |\n')
        f.write('| ' + ' | '.join(['---'] * len(headers)) + ' |\n')
        for row in rows:
            vals = [
                f"{row[m]*100:.2f}" if row[m] is not None else 'NA'
                for m in METRIC_COLS
            ]
            f.write(f"| {row['experiment']} | {row['family']} | " + ' | '.join(vals) + ' |\n')


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='ViT-GPT2 image captioning — regular fine-tuning')

    p.add_argument('--strategy', choices=list(STRATEGIES), default='pretrained')
    p.add_argument('--encoder-mode', choices=ENCODER_MODES, default=None,
                   help='Override encoder fine-tuning mode from strategy default')
    p.add_argument('--decoder-mode', choices=DECODER_MODES, default=None,
                   help='Override decoder fine-tuning mode from strategy default')
    p.add_argument('--encoder-ft-layers', type=int, default=ENCODER_FT_LAYERS,
                   metavar='N',
                   help=f'ViT blocks to unfreeze from top (last_n mode). Default: {ENCODER_FT_LAYERS}')
    p.add_argument('--decoder-ft-layers', type=int, default=DECODER_FT_LAYERS,
                   metavar='N',
                   help=f'GPT-2 blocks to unfreeze from top (last_n mode). Default: {DECODER_FT_LAYERS}')
    p.add_argument('--lr', type=float, default=LR,
                   help=f'Learning rate. Default: {LR}')
    p.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY,
                   help=f'AdamW weight decay. Default: {WEIGHT_DECAY}')
    p.add_argument('--label-smoothing', type=float, default=LABEL_SMOOTHING,
                   help=f'Label smoothing coefficient. Default: {LABEL_SMOOTHING}')
    p.add_argument('--grad-clip', type=float, default=GRAD_CLIP,
                   help=f'Gradient clipping max-norm (0 = disabled). Default: {GRAD_CLIP}')
    p.add_argument('--repetition-penalty', type=float, default=REPETITION_PENALTY,
                   help=f'Generation repetition penalty (1.0 = disabled). Default: {REPETITION_PENALTY}')
    p.add_argument('--epochs', type=int, default=EPOCHS,
                   help=f'Number of training epochs. Default: {EPOCHS}')
    p.add_argument('--resume-from', type=str, default=None, metavar='CHECKPOINT',
                   help='Path to a .pt checkpoint to load before training starts '
                        '(overrides the automatic same-experiment lookup)')
    p.add_argument('--report-only', action='store_true',
                   help='Skip training/evaluation; only regenerate the report table')
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    if args.report_only:
        write_experiment_report(RESULTS_ROOT)
        print(f"Experiment report updated in: {RESULTS_ROOT}")
        return

    # Allow per-run override of encoder/decoder mode
    strategy_cfg  = STRATEGIES[args.strategy]
    encoder_mode  = args.encoder_mode or strategy_cfg['encoder_mode']
    decoder_mode  = args.decoder_mode or strategy_cfg['decoder_mode']

    dataset_paths  = get_dataset_paths(DATASET_ROOT)
    train_samples  = load_vizwiz_split(dataset_paths['train_json'])
    val_samples    = load_vizwiz_split(dataset_paths['val_json'])

    print(f"Device: {device}", flush=True)
    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}", flush=True)

    image_processor = AutoImageProcessor.from_pretrained(HF_MODEL_NAME)
    tokenizer       = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dl_train = make_dataloader(
        train_samples, dataset_paths['train_img_path'],
        shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        image_processor=image_processor, tokenizer=tokenizer,
        max_target_length=MAX_TARGET_LENGTH, device=device,
    )
    dl_val = make_dataloader(
        val_samples, dataset_paths['val_img_path'],
        shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        image_processor=image_processor, tokenizer=tokenizer,
        max_target_length=MAX_TARGET_LENGTH, device=device,
    )

    model = VisionEncoderDecoderCaptioningModel(
        pretrained_model_name=HF_MODEL_NAME,
        encoder_mode=encoder_mode,
        decoder_mode=decoder_mode,
        encoder_ft_layers=args.encoder_ft_layers,
        decoder_ft_layers=args.decoder_ft_layers,
    ).to(device)

    model.configure_generation(
        tokenizer,
        max_length=MAX_TARGET_LENGTH,
        num_beams=NUM_BEAMS,
        repetition_penalty=args.repetition_penalty,
    )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_trainable:,} / {n_total:,}", flush=True)
    print(f"  Encoder mode : {encoder_mode}" +
          (f" (last {args.encoder_ft_layers} layers)" if encoder_mode == 'last_n' else ""),
          flush=True)
    print(f"  Decoder mode : {decoder_mode}" +
          (f" (last {args.decoder_ft_layers} layers)" if decoder_mode == 'last_n' else ""),
          flush=True)
    print(f"  LR={args.lr}  WD={args.weight_decay}  LS={args.label_smoothing}"
          f"  GradClip={args.grad_clip}  RepPenalty={args.repetition_penalty}",
          flush=True)

    # Build an experiment name that captures the key choices
    enc_tag = encoder_mode if encoder_mode != 'last_n' else f'enc{args.encoder_ft_layers}L'
    dec_tag = decoder_mode if decoder_mode != 'last_n' else f'dec{args.decoder_ft_layers}L'
    experiment_name = f"vit_gpt2_{args.strategy}_{enc_tag}_{dec_tag}"
    results_dir, _, best_weights = build_results_dir(RESULTS_ROOT, experiment_name)

    # Load checkpoint: explicit --resume-from takes priority, then same-experiment auto-lookup
    ckpt_to_load = None
    if args.resume_from:
        ckpt_to_load = args.resume_from
    elif not strategy_cfg['eval_only'] and os.path.exists(best_weights):
        ckpt_to_load = best_weights
    if ckpt_to_load:
        print(f"Loading weights from: {ckpt_to_load}", flush=True)
        model.load_state_dict(torch.load(ckpt_to_load, map_location=device))

    generation_kwargs = {
        'max_length':         MAX_TARGET_LENGTH,
        'num_beams':          NUM_BEAMS,
        'repetition_penalty': args.repetition_penalty,
    }
    metric = Metric()

    if strategy_cfg['eval_only']:
        val_loss, best_metrics, best_preds = eval_hf_epoch(
            model=model,
            metric=metric,
            dataloader=dl_val,
            tokenizer=tokenizer,
            device=device,
            generation_kwargs=generation_kwargs,
        )
        best_epoch = -1
        history    = []
        print(f"val loss: {val_loss:.4f}", flush=True)
        if best_metrics:
            for k, v in best_metrics.items():
                print(f"  {k}: {v*100:.2f}%", flush=True)
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        # ReduceLROnPlateau: halve LR if val loss does not improve for 3 epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        history, best_metrics, best_preds, best_epoch = train_hf(
            model=model,
            optimizer=optimizer,
            metric=metric,
            dataloader_train=dl_train,
            dataloader_val=dl_val,
            tokenizer=tokenizer,
            epochs=args.epochs,
            patience=PATIENCE,
            device=device,
            save_path=best_weights,
            min_delta=MIN_DELTA,
            label_smoothing=args.label_smoothing,
            grad_clip=args.grad_clip,
            generation_kwargs=generation_kwargs,
            viz_batch=None,   # set to a real batch to enable per-epoch visualizations
            scheduler=scheduler,
        )

    summary = {
        'family':              'vit-gpt2',
        'experiment':          experiment_name,
        'label':               strategy_cfg['label'],
        'hf_model_name':       HF_MODEL_NAME,
        'best_epoch':          best_epoch,
        'num_trainable_params': n_trainable,
        'peft_method':         'none',
        'encoder_mode':        encoder_mode,
        'decoder_mode':        decoder_mode,
        'encoder_ft_layers':   args.encoder_ft_layers,
        'decoder_ft_layers':   args.decoder_ft_layers,
        'lr':                  args.lr,
        'weight_decay':        args.weight_decay,
        'label_smoothing':     args.label_smoothing,
        'grad_clip':           args.grad_clip,
        'repetition_penalty':  args.repetition_penalty,
        **{f'val_{k}': v for k, v in (best_metrics or {}).items()},
    }

    save_history_csv(history, os.path.join(results_dir, 'history.csv'))
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(results_dir, 'predictions_val.json'), 'w') as f:
        json.dump(best_preds, f, indent=2)

    write_experiment_report(RESULTS_ROOT)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Best epoch : {best_epoch}")
    if best_metrics:
        for k, v in best_metrics.items():
            print(f"  {k}: {v*100:.2f}%")
    print(f"\nResults saved to : {results_dir}")
    if history:
        print(f"Best weights     : {best_weights}")
    print(f"Report table     : {os.path.join(RESULTS_ROOT, 'experiment_report.md')}")


if __name__ == '__main__':
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    main()
