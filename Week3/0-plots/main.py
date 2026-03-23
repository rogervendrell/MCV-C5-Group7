"""
Generate training plots from one or more SLURM .out log files.

Usage examples
--------------
# Single run – all plots saved to ./figures/
python main.py path/to/run.out

# Compare two runs with custom labels
python main.py run_a.out run_b.out --labels "Baseline" "Word-level"

# Only specific metrics
python main.py run.out --metrics loss bleu1 bleu2

# Custom output directory
python main.py run.out --output /path/to/plots/
"""

import argparse
import sys
from pathlib import Path

from parse_out import parse_out
from plots import ALL_METRIC_PLOTS, plot_all


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot training metrics from SLURM .out files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "out_files",
        nargs="+",
        metavar="FILE.out",
        help="One or more .out log files to plot.",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        metavar="LABEL",
        default=None,
        help="Display labels for each run (must match number of files).",
    )
    p.add_argument(
        "--output", "-o",
        default="figures",
        metavar="DIR",
        help="Output directory for PNG files (default: ./figures/).",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        choices=[name for name, _ in ALL_METRIC_PLOTS],
        default=None,
        metavar="METRIC",
        help=(
            "Metrics to plot. Choices: "
            + ", ".join(name for name, _ in ALL_METRIC_PLOTS)
            + ". Default: all."
        ),
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    # Validate --labels count
    if args.labels is not None and len(args.labels) != len(args.out_files):
        print(
            f"Error: --labels has {len(args.labels)} entries "
            f"but {len(args.out_files)} files were given.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Parse runs
    runs = []
    for i, path_str in enumerate(args.out_files):
        path = Path(path_str)
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        label = args.labels[i] if args.labels else None
        run = parse_out(path, label=label)
        runs.append(run)
        print(
            f"Parsed '{run.label}': {len(run.epochs)} epoch(s)"
            + (f", vocab {run.vocab_size}" if run.vocab_size else "")
            + (f", {run.trainable_params:,} params" if run.trainable_params else "")
        )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Select which plots to produce
    metric_map = {name: fn for name, fn in ALL_METRIC_PLOTS}
    selected = args.metrics if args.metrics else list(metric_map)

    print(f"\nWriting plots to: {out_dir.resolve()}")
    for name in selected:
        metric_map[name](runs, out_dir / f"{name}.png")


if __name__ == "__main__":
    main()
