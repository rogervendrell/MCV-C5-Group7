"""Entry point for image-captioning experiments (ResNet + Transformer decoder)."""
import json
import os

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import (
    VizWizDataset,
    load_vizwiz_split,
    collate_fn,
    TRAIN_IMG_PATH,
    VAL_IMG_PATH,
    TRAIN_JSON,
    VAL_JSON,
)
from vocabulary import Vocabulary
from metrics import Metric
from model import BaselineModel
from train import train, eval_epoch

# Shared hyper-parameters
EPOCHS      = 10
BATCH_SIZE  = 32
NUM_WORKERS = 4
LR          = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE    = 5
MIN_DELTA   = 1e-4
NUM_DECODER_LAYERS = 2
HIDDEN_SIZE = 1024
DROPOUT     = 0.3

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

_HERE   = os.path.dirname(os.path.abspath(__file__))
_JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')

# Two experiment configurations to run sequentially
EXPERIMENTS = [
    {
        'name':                'resnet18_transformer_frozen',
        'encoder_name':        'microsoft/resnet-18',
        'freeze_encoder':      True,
        'unfreeze_last_stage': False,
    },
    {
        'name':                'resnet34_transformer_unfreeze_last',
        'encoder_name':        'microsoft/resnet-34',
        'freeze_encoder':      True,
        'unfreeze_last_stage': True,
    },
]


def make_dataloader(samples, img_path, vocab, shuffle):
    ds = VizWizDataset(samples, img_path, vocab=vocab)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == 'cuda'),
        persistent_workers=(NUM_WORKERS > 0),
        collate_fn=collate_fn,
    )


def run_experiment(cfg, dl_train, dl_val, vocab):
    results_dir = os.path.join(_HERE, 'results', _JOB_ID, cfg['name'])
    weights_dir = os.path.join(results_dir, 'weights')
    best_weights = os.path.join(weights_dir, 'best_model.pt')
    os.makedirs(weights_dir, exist_ok=True)

    print(f"\n{'='*60}", flush=True)
    print(f"EXPERIMENT: {cfg['name']}", flush=True)
    print(f"  encoder        : {cfg['encoder_name']}", flush=True)
    print(f"  freeze_encoder : {cfg['freeze_encoder']}", flush=True)
    print(f"  unfreeze_last  : {cfg['unfreeze_last_stage']}", flush=True)
    print(f"  hidden_size    : {HIDDEN_SIZE}", flush=True)
    print(f"  dropout        : {DROPOUT}", flush=True)
    print(f"  lr             : {LR}  weight_decay: {WEIGHT_DECAY}", flush=True)
    print(f"{'='*60}", flush=True)

    model = BaselineModel(
        vocab_size=len(vocab),
        sos_idx=vocab.word2idx['<SOS>'],
        freeze_encoder=cfg['freeze_encoder'],
        num_decoder_layers=NUM_DECODER_LAYERS,
        encoder_name=cfg['encoder_name'],
        unfreeze_last_stage=cfg['unfreeze_last_stage'],
        hidden_size=HIDDEN_SIZE,
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<PAD>'])
    metric    = Metric()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history, best_metrics, best_preds, best_epoch = train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        metric=metric,
        dataloader_train=dl_train,
        dataloader_val=dl_val,
        epochs=EPOCHS,
        patience=PATIENCE,
        device=DEVICE,
        save_path=best_weights,
        vocab=vocab,
        min_delta=MIN_DELTA,
    )

    # Save results
    if history:
        pd.DataFrame(history).to_csv(os.path.join(results_dir, 'history.csv'), index=False)

    summary = {
        'model':        cfg['name'],
        'encoder':      cfg['encoder_name'],
        'hidden_size':  HIDDEN_SIZE,
        'dropout':      DROPOUT,
        'lr':           LR,
        'weight_decay': WEIGHT_DECAY,
        'best_epoch':   best_epoch,
        'num_params':   n_params,
        **{f'val_{k}': v for k, v in (best_metrics or {}).items()},
    }
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(results_dir, 'predictions_val.json'), 'w') as f:
        json.dump(best_preds, f, indent=2)

    print(f"\nFINAL RESULTS — {cfg['name']}")
    print(f"Best epoch : {best_epoch}")
    if best_metrics:
        for k, v in best_metrics.items():
            print(f"  {k}: {v*100:.2f}%")
    print(f"Results saved to: {results_dir}")
    print(f"Best weights   : {best_weights}")


def main():
    print(f"Device: {DEVICE}", flush=True)

    # Build shared vocab and dataloaders once
    train_samples = load_vizwiz_split(TRAIN_JSON)
    val_samples   = load_vizwiz_split(VAL_JSON)

    all_train_captions = [cap for item in train_samples for cap in item['captions']]
    vocab = Vocabulary(freq_threshold=2)
    vocab.build_vocabulary(all_train_captions)
    print(f"Vocab size: {len(vocab)}", flush=True)
    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}", flush=True)

    dl_train = make_dataloader(train_samples, TRAIN_IMG_PATH, vocab, shuffle=True)
    dl_val   = make_dataloader(val_samples,   VAL_IMG_PATH, vocab, shuffle=False)

    for cfg in EXPERIMENTS:
        run_experiment(cfg, dl_train, dl_val, vocab)


if __name__ == '__main__':
    main()
