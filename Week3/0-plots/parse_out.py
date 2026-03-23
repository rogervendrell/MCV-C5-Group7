"""
Parser for SLURM .out training log files.

Each .out file corresponds to one training run. The parser extracts
per-epoch train/val loss and evaluation metrics (BLEU-1/2, ROUGE-L, METEOR).
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EpochData:
    epoch: int
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    bleu1: Optional[float] = None   # stored in [0, 1]
    bleu2: Optional[float] = None
    rouge_l: Optional[float] = None
    meteor: Optional[float] = None


@dataclass
class RunData:
    label: str
    path: Path
    epochs: list[EpochData] = field(default_factory=list)
    # metadata from header lines
    vocab_size: Optional[int] = None
    trainable_params: Optional[int] = None
    train_samples: Optional[int] = None
    val_samples: Optional[int] = None

    def get_metric(self, name: str) -> tuple[list[int], list[float]]:
        """Return (epoch_indices, values) for a given metric name.

        name is one of: train_loss, val_loss, bleu1, bleu2, rouge_l, meteor
        Only epochs that have a value for that metric are returned.
        """
        xs, ys = [], []
        for e in self.epochs:
            v = getattr(e, name, None)
            if v is not None:
                xs.append(e.epoch)
                ys.append(v)
        return xs, ys


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_RE_TRAIN_LOSS = re.compile(r'^\[epoch (\d+)\] train loss: ([0-9.]+)')
_RE_VAL_LOSS   = re.compile(
    r'^\[epoch (\d+)\] val loss: ([0-9.]+)'
    r' \| BLEU-1:([0-9.]+)% BLEU-2:([0-9.]+)%'
    r' ROUGE-L:([0-9.]+)% METEOR:([0-9.]+)%'
)
_RE_VOCAB      = re.compile(r'Vocab size is (\d+)')
_RE_PARAMS     = re.compile(r'Trainable parameters: ([\d,]+)')
_RE_SAMPLES    = re.compile(r'Train samples: (\d+), Val samples: (\d+)')


def parse_out(path: str | Path, label: Optional[str] = None) -> RunData:
    """Parse a single .out file and return a RunData object."""
    path = Path(path)
    if label is None:
        label = path.stem

    run = RunData(label=label, path=path)
    epochs: dict[int, EpochData] = {}

    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()

            m = _RE_VOCAB.search(line)
            if m:
                run.vocab_size = int(m.group(1))
                continue

            m = _RE_PARAMS.search(line)
            if m:
                run.trainable_params = int(m.group(1).replace(',', ''))
                continue

            m = _RE_SAMPLES.search(line)
            if m:
                run.train_samples = int(m.group(1))
                run.val_samples   = int(m.group(2))
                continue

            m = _RE_TRAIN_LOSS.match(line)
            if m:
                epoch = int(m.group(1))
                epochs.setdefault(epoch, EpochData(epoch=epoch))
                epochs[epoch].train_loss = float(m.group(2))
                continue

            m = _RE_VAL_LOSS.match(line)
            if m:
                epoch = int(m.group(1))
                epochs.setdefault(epoch, EpochData(epoch=epoch))
                e = epochs[epoch]
                e.val_loss = float(m.group(2))
                e.bleu1    = float(m.group(3)) / 100.0
                e.bleu2    = float(m.group(4)) / 100.0
                e.rouge_l  = float(m.group(5)) / 100.0
                e.meteor   = float(m.group(6)) / 100.0
                continue

    run.epochs = [epochs[k] for k in sorted(epochs)]
    return run
