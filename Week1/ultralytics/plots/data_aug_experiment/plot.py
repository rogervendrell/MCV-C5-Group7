import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_augmentation_comparison(csv_path, output_dir):
    df = pd.read_csv(csv_path)

    def f1(p, r):
        return (2 * p * r) / (p + r) if (p + r) != 0 else 0

    df["f1_score"] = df.apply(
        lambda row: f1(row["precision"], row["recall"]),
        axis=1
    )

    metrics = ["mAP50", "mAP50-95", "precision", "recall", "f1_score"]

    df_no_aug = df[df["augmentation"] == False]
    df_aug = df[df["augmentation"] == True]

    if df_no_aug.empty or df_aug.empty:
        raise ValueError("CSV must contain both augmentation=True and False rows.")

    values_no_aug = df_no_aug.iloc[0][metrics].values
    values_aug = df_aug.iloc[0][metrics].values

    x = np.arange(len(metrics))
    width = 0.35

    color_no_aug = "#A8C5DA"
    color_aug = "#F6C28B"

    fig, ax = plt.subplots(figsize=(7, 6))

    bars1 = ax.bar(
        x - width/2, values_no_aug, width,
        label="No Augmentation",
        color=color_no_aug
    )
    bars2 = ax.bar(
        x + width/2, values_aug, width,
        label="With Augmentation",
        color=color_aug
    )

    ax.set_ylabel("Score")
    ax.set_title("Comparison: With vs Without Data Augmentation")
    ax.set_xticks(x)
    ax.set_xticklabels(["mAP50", "mAP50-95", "Precision", "Recall", "F1 Score"], rotation=30)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.1)) 
    ax.legend()

    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "augmentation_comparison.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


csv_path = "/ghome/group07/MCV-C5-Group7/ultralytics/plots/data_aug_experiment/data.csv"
output_dir = "/ghome/group07/MCV-C5-Group7/ultralytics/plots/data_aug_experiment/"

plot_path = plot_augmentation_comparison(csv_path, output_dir)
print("Plot saved to:", plot_path)