"""
Qualitative comparison of caption predictions across multiple models.

Usage:
    python qualitative.py \
        --predictions path/to/pred1.json "Model A" \
        --predictions path/to/pred2.json "Model B"

Each --predictions argument takes two values: the JSON file path and a display name.
All prediction files must share the same image ordering (same val split).
Edit SELECTED_IMAGES below to choose which images to compare.
"""

import argparse
import json

# ── Fixed images to compare ───────────────────────────────────────────────────
SELECTED_IMAGES = [
    "VizWiz_val_00003402.jpg",
    "VizWiz_val_00000911.jpg",
    "VizWiz_val_00003395.jpg",
]
# ─────────────────────────────────────────────────────────────────────────────


def load_predictions(path: str) -> dict[str, dict]:
    with open(path) as f:
        data = json.load(f)
    return {entry["image"]: entry for entry in data}


def print_comparison(selected: list[str], models: list[tuple[str, dict]]):
    sep = "=" * 70
    for image in selected:
        print(sep)
        print(f"IMAGE: {image}")
        print()

        # Ground truth references (same across models — take from first)
        refs = models[0][1][image]["references"]
        print("REFERENCES:")
        for i, ref in enumerate(refs, 1):
            print(f"  {i}. {ref}")
        print()

        for name, preds in models:
            prediction = preds[image]["prediction"]
            print(f"[{name}]")
            print(f"  {prediction}")
            print()

    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Qualitative caption comparison")
    parser.add_argument(
        "--predictions",
        nargs=2,
        metavar=("JSON_PATH", "MODEL_NAME"),
        action="append",
        required=True,
        help="Path to predictions_val.json and a display name (repeatable)",
    )
    args = parser.parse_args()

    models: list[tuple[str, dict]] = []
    for path, name in args.predictions:
        preds = load_predictions(path)
        models.append((name, preds))
        print(f"Loaded {len(preds)} predictions for '{name}' from {path}")

    # Warn if any model is missing images that others have
    image_sets = [set(preds.keys()) for _, preds in models]
    all_images = set.union(*image_sets)
    common_images = set.intersection(*image_sets)
    if len(common_images) < len(all_images):
        print(f"WARNING: validation sets are not identical across models!")
        for (name, _), img_set in zip(models, image_sets):
            missing = all_images - img_set
            if missing:
                print(f"  '{name}' is missing {len(missing)} image(s): {sorted(missing)}")
        print(f"Proceeding with {len(common_images)} common images.")
    else:
        print(f"All models share the same {len(common_images)} images.")

    missing_selected = [img for img in SELECTED_IMAGES if img not in common_images]
    if missing_selected:
        print(f"WARNING: these selected images are not in the predictions: {missing_selected}")
    selected = [img for img in SELECTED_IMAGES if img in common_images]
    print(f"Comparing {len(selected)} images\n")

    print_comparison(selected, models)


if __name__ == "__main__":
    main()
