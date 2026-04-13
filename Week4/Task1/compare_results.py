import argparse
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from dataset import get_dataset_paths
from visualization import (
    plot_learning_curves,
    plot_metric_distributions,
    plot_radar_chart,
    plot_caption_length_analysis,
    plot_vocabulary_analysis,
    plot_best_worst_gallery,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_RESULTS_ROOT = os.path.join(_HERE, 'results')
DEFAULT_OUTPUT_DIR   = os.path.join(_HERE, 'post_hoc_plots')
ALL_PLOTS = [
    'learning_curves',
    'metric_distributions',
    'radar',
    'caption_lengths',
    'vocabulary',
    'gallery',
]


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def discover_experiments(results_root: str) -> list[dict]:
    pattern  = os.path.join(results_root, '**', 'summary.json')
    found    = sorted(glob.glob(pattern, recursive=True))
    complete = []

    for summary_path in found:
        exp_dir          = os.path.dirname(summary_path)
        history_path     = os.path.join(exp_dir, 'history.csv')
        predictions_path = os.path.join(exp_dir, 'predictions_val.json')

        with open(summary_path) as f:
            summary = json.load(f)

        if summary.get('family') != 'vit-gpt2':
            continue

        complete.append({
            'label':            summary.get('label', summary.get('experiment', exp_dir)),
            'summary_path':     summary_path,
            'history_path':     history_path if os.path.exists(history_path) else None,
            'predictions_path': predictions_path if os.path.exists(predictions_path) else None,
            'eval_only':        not os.path.exists(history_path),
        })

    return complete


def build_dicts(experiments: list[dict], key: str) -> dict:
    """Build a {label: path} dict for the given file key, skipping missing files."""
    return {
        e['label']: e[key]
        for e in experiments
        if e.get(key) is not None
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Post-hoc comparison plots for captioning experiments')
    p.add_argument('--results-root', default=DEFAULT_RESULTS_ROOT, metavar='DIR',
                   help=f'Root of the results tree. Default: {DEFAULT_RESULTS_ROOT}')
    p.add_argument('--image-dir', default=None, metavar='DIR',
                   help='Directory containing raw val images (for gallery). '
                        'Defaults to the val/ folder reported by dataset.py.')
    p.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, metavar='DIR',
                   help=f'Where to save all comparison plots. Default: {DEFAULT_OUTPUT_DIR}')
    p.add_argument('--gallery-n', type=int, default=5, metavar='N',
                   help='Number of best and worst samples in the gallery. Default: 5')
    p.add_argument('--gallery-metric', default='METEOR',
                   choices=['BLEU-1', 'ROUGE-L', 'METEOR'],
                   help='Metric used to rank samples for the gallery. Default: METEOR')
    p.add_argument('--only', default=None, metavar='PLOTS',
                   help='Comma-separated subset of plots to generate. '
                        f'Choices: {", ".join(ALL_PLOTS)}')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    wanted = ALL_PLOTS
    if args.only:
        wanted = [p.strip() for p in args.only.split(',')]
        invalid = set(wanted) - set(ALL_PLOTS)
        if invalid:
            print(f'[compare] Unknown plot names: {invalid}')
            print(f'[compare] Valid choices: {ALL_PLOTS}')
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover experiments
    experiments = discover_experiments(args.results_root)
    if not experiments:
        print(f'[compare] No completed vit-gpt2 experiments found under {args.results_root}')
        sys.exit(0)

    print(f'[compare] Found {len(experiments)} experiment(s):')
    for e in experiments:
        flag = '(eval-only)' if e['eval_only'] else ''
        print(f'  • {e["label"]} {flag}')

    # Build path dicts
    histories    = build_dicts(experiments, 'history_path')
    predictions  = build_dicts(experiments, 'predictions_path')
    summaries    = build_dicts(experiments, 'summary_path')

    # Image directory (for gallery)
    image_dir = args.image_dir
    if image_dir is None:
        try:
            image_dir = get_dataset_paths()['val_img_path']
        except Exception:
            image_dir = ''
    if not os.path.isdir(image_dir):
        print(f'[compare] Warning: image_dir "{image_dir}" not found — gallery will be skipped.')
        image_dir = None

    # Generate plots
    out = args.output_dir

    if 'learning_curves' in wanted:
        if histories:
            print('[compare] Plotting learning curves ...')
            plot_learning_curves(histories, output_dir=out)
        else:
            print('[compare] Skipping learning_curves — no history.csv files found '
                  '(pretrained-only runs have no training history).')

    if 'metric_distributions' in wanted:
        if predictions:
            print('[compare] Plotting metric distributions ...')
            plot_metric_distributions(predictions, output_dir=out)
        else:
            print('[compare] Skipping metric_distributions — no predictions_val.json found.')

    if 'radar' in wanted:
        if summaries:
            print('[compare] Plotting radar chart ...')
            plot_radar_chart(summaries, output_dir=out)
        else:
            print('[compare] Skipping radar — no summary.json files found.')

    if 'caption_lengths' in wanted:
        if predictions:
            print('[compare] Plotting caption length analysis ...')
            plot_caption_length_analysis(predictions, output_dir=out)
        else:
            print('[compare] Skipping caption_lengths — no predictions_val.json found.')

    if 'vocabulary' in wanted:
        if predictions:
            print('[compare] Plotting vocabulary analysis ...')
            plot_vocabulary_analysis(predictions, output_dir=out)
        else:
            print('[compare] Skipping vocabulary — no predictions_val.json found.')

    if 'gallery' in wanted:
        if image_dir and predictions:
            # Produce a gallery for each experiment separately so you can
            # compare qualitative outputs side by side.
            for label, pred_path in predictions.items():
                safe_label = label.replace(' ', '_').replace('|', '-')
                gallery_dir = os.path.join(out, f'gallery_{safe_label}')
                print(f'[compare] Plotting gallery for "{label}" ...')
                plot_best_worst_gallery(
                    predictions_json=pred_path,
                    image_dir=image_dir,
                    output_dir=gallery_dir,
                    n=args.gallery_n,
                    sort_metric=args.gallery_metric,
                )
        else:
            reason = 'no predictions_val.json found' if not predictions \
                else 'image_dir not available'
            print(f'[compare] Skipping gallery — {reason}.')

    print(f'\n[compare] Done. All plots saved to: {out}')


if __name__ == '__main__':
    main()
