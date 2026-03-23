"""
Plotting functions for training run comparisons.

Each function accepts a list of RunData objects so that multiple runs
can be overlaid in the same figure.  Call with a single-element list
when you just want to inspect one run.
"""

from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from parse_out import RunData


# ---------------------------------------------------------------------------
# Palette & style
# ---------------------------------------------------------------------------

# Okabe–Ito colour-blind-safe palette, extended with a few extras
_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # amber
    "#009E73",  # green
    "#CC79A7",  # mauve
    "#D55E00",  # vermilion
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#999999",  # grey
]

_LINE_STYLE_CYCLE = ["-", "--", "-.", ":"]


def _apply_base_style() -> None:
    """Apply a clean, minimal rcParams style."""
    mpl.rcParams.update({
        # Figure
        "figure.facecolor":     "white",
        "figure.dpi":           130,
        "figure.figsize":       (6.5, 4.2),
        # Axes
        "axes.facecolor":       "#f9f9f9",
        "axes.edgecolor":       "#cccccc",
        "axes.linewidth":       0.8,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.grid":            True,
        "axes.axisbelow":       True,
        # Grid
        "grid.color":           "#e0e0e0",
        "grid.linewidth":       0.7,
        "grid.linestyle":       "--",
        # Lines
        "lines.linewidth":      2.0,
        "lines.markersize":     5,
        # Font
        "font.family":          "sans-serif",
        "font.size":            11,
        "axes.titlesize":       12,
        "axes.titleweight":     "semibold",
        "axes.labelsize":       10,
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        # Legend
        "legend.framealpha":    0.9,
        "legend.edgecolor":     "#cccccc",
        "legend.fontsize":      9,
        "legend.borderpad":     0.6,
        # Saving
        "savefig.bbox":         "tight",
        "savefig.facecolor":    "white",
    })


def _new_fig(title: str, ylabel: str) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    return fig, ax


def _save(fig: mpl.figure.Figure, out_path: Path) -> None:
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


def _color(run_idx: int) -> str:
    return _PALETTE[run_idx % len(_PALETTE)]


def _linestyle(run_idx: int) -> str:
    return _LINE_STYLE_CYCLE[run_idx % len(_LINE_STYLE_CYCLE)]


# ---------------------------------------------------------------------------
# Individual plot helpers
# ---------------------------------------------------------------------------

def plot_loss(
    runs: list[RunData],
    out_path: Path,
    *,
    show_train: bool = True,
    show_val: bool = True,
) -> None:
    """Train + val loss curves for one or more runs."""
    _apply_base_style()
    title = "Loss" if len(runs) > 1 else f"Loss — {runs[0].label}"
    fig, ax = _new_fig(title, "Cross-entropy loss")

    for i, run in enumerate(runs):
        color = _color(i)
        ls    = _linestyle(i)
        xs_tr, ys_tr = run.get_metric("train_loss")
        xs_vl, ys_vl = run.get_metric("val_loss")

        if show_train and xs_tr:
            label = f"{run.label} – train" if len(runs) > 1 else "train"
            ax.plot(xs_tr, ys_tr, color=color, linestyle=ls,
                    marker="o", markevery=max(1, len(xs_tr)//10),
                    label=label, alpha=0.85)

        if show_val and xs_vl:
            label = f"{run.label} – val" if len(runs) > 1 else "val"
            ax.plot(xs_vl, ys_vl, color=color, linestyle=ls,
                    marker="s", markevery=max(1, len(xs_vl)//10),
                    linewidth=1.4, alpha=0.65, label=label)

    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    if ax.get_lines():
        ax.legend()
    _save(fig, out_path)


def _plot_metric(
    runs: list[RunData],
    metric_key: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """Generic helper for a single [0, 1]-bounded evaluation metric."""
    _apply_base_style()
    fig, ax = _new_fig(title, ylabel)

    for i, run in enumerate(runs):
        xs, ys = run.get_metric(metric_key)
        if not xs:
            continue
        label = run.label if len(runs) > 1 else None
        ax.plot(xs, ys,
                color=_color(i),
                linestyle=_linestyle(i),
                marker="o",
                markevery=max(1, len(xs)//10),
                label=label)

    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    if len(runs) > 1 and ax.get_lines():
        ax.legend()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Public per-metric functions
# ---------------------------------------------------------------------------

def plot_bleu1(runs: list[RunData], out_path: Path) -> None:
    _plot_metric(runs, "bleu1", "BLEU-1", "BLEU-1", out_path)


def plot_bleu2(runs: list[RunData], out_path: Path) -> None:
    _plot_metric(runs, "bleu2", "BLEU-2", "BLEU-2", out_path)


def plot_rouge_l(runs: list[RunData], out_path: Path) -> None:
    _plot_metric(runs, "rouge_l", "ROUGE-L", "ROUGE-L", out_path)


def plot_meteor(runs: list[RunData], out_path: Path) -> None:
    _plot_metric(runs, "meteor", "METEOR", "METEOR", out_path)


# ---------------------------------------------------------------------------
# Convenience: produce all plots at once
# ---------------------------------------------------------------------------

ALL_METRIC_PLOTS = [
    ("loss",    plot_loss),
    ("bleu1",   plot_bleu1),
    ("bleu2",   plot_bleu2),
    ("rouge_l", plot_rouge_l),
    ("meteor",  plot_meteor),
]


def plot_all(runs: list[RunData], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in ALL_METRIC_PLOTS:
        fn(runs, out_dir / f"{name}.png")
