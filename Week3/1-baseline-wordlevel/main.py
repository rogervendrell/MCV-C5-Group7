"""Entry point for the baseline image-captioning model (ResNet-18 + GRU)."""
import argparse
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

# Hyper-parameters
EPOCHS      = 10
BATCH_SIZE  = 32
NUM_WORKERS = 4
LR          = 1e-3
PATIENCE    = 5
MIN_DELTA   = 1e-4
FREEZE_ENCODER  = True
NUM_DECODER_LAYERS = 1

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Output paths
_HERE        = os.path.dirname(os.path.abspath(__file__))
_JOB_ID      = os.environ.get('SLURM_JOB_ID', 'local')
RESULTS_DIR  = os.path.join(_HERE, 'results', _JOB_ID)
WEIGHTS_DIR  = os.path.join(RESULTS_DIR, 'weights')
BEST_WEIGHTS = os.path.join(WEIGHTS_DIR, 'best_model.pt')

os.makedirs(WEIGHTS_DIR, exist_ok=True)


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


def parse_args():
    parser = argparse.ArgumentParser(description='ResNet-18 + GRU image captioning baseline')
    parser.add_argument(
        '--load-weights', metavar='PATH', default=None,
        help='Path to a .pt weights file to load before training or evaluation',
    )
    parser.add_argument(
        '--eval-only', action='store_true',
        help='Skip training and run one evaluation pass on the validation set',
    )
    parser.add_argument(
        '--num-decoder-layers', type=int, default=None, metavar='N',
        help='Number of GRU layers in the decoder (overrides NUM_DECODER_LAYERS constant)',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Device: {DEVICE}", flush=True)

    # Data
    train_samples = load_vizwiz_split(TRAIN_JSON)
    val_samples   = load_vizwiz_split(VAL_JSON)

    # Extract all captions to build vocab
    all_train_captions = [cap for item in train_samples for cap in item['captions']]
    vocab = Vocabulary(freq_threshold=2) 
    vocab.build_vocabulary(all_train_captions)
    print(f"DEBUG: Vocab size is {len(vocab)}")
    
    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}", flush=True)

    dl_train = make_dataloader(train_samples, TRAIN_IMG_PATH, vocab, shuffle=True)
    dl_val   = make_dataloader(val_samples,   VAL_IMG_PATH, vocab, shuffle=False)

    # Model
    print(f"DEBUG: Attempting to initialize model with vocab_size: {len(vocab)}")
    num_layers = args.num_decoder_layers if args.num_decoder_layers is not None else NUM_DECODER_LAYERS
    model     = BaselineModel(vocab_size=len(vocab), sos_idx=vocab.word2idx['<SOS>'], freeze_encoder=FREEZE_ENCODER, num_decoder_layers=num_layers).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<PAD>'])
    metric    = Metric()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}", flush=True)

    # Optionally load weights
    if args.load_weights:
        print(f"Loading weights from: {args.load_weights}", flush=True)
        state = torch.load(args.load_weights, map_location=DEVICE)
        model.load_state_dict(state)

    if args.eval_only:
        # Single evaluation pass — no training, no checkpoint saved
        val_loss, best_metrics, best_preds = eval_epoch(
            model, criterion, metric, dl_val, DEVICE,
        )
        best_epoch = -1
        history = []
        print(f"val loss: {val_loss:.4f}", flush=True)
    else:
        # Full training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
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
            save_path=BEST_WEIGHTS,
            vocab=vocab,
            min_delta=MIN_DELTA,
        )

    # Save results
    if history:
        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(RESULTS_DIR, 'history.csv'), index=False)

    summary = {
        'model':       'ResNet18_GRU_baseline',
        'best_epoch':  best_epoch,
        'num_params':  n_params,
        **{f'val_{k}': v for k, v in (best_metrics or {}).items()},
    }
    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(RESULTS_DIR, 'predictions_val.json'), 'w') as f:
        json.dump(best_preds, f, indent=2)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Best epoch : {best_epoch}")
    if best_metrics:
        for k, v in best_metrics.items():
            print(f"  {k}: {v*100:.2f}%")
    print(f"\nResults saved to: {RESULTS_DIR}")
    if not args.eval_only:
        print(f"Best weights   : {BEST_WEIGHTS}")


if __name__ == '__main__':
    main()
