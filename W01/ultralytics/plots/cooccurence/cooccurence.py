import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def compute_distribution(labels_dir):
    """
    Compute (num_pedestrians, num_cars) frequency distribution
    for a given labels directory.
    """
    distribution = defaultdict(int)

    for filename in os.listdir(labels_dir):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(labels_dir, filename)

        num_pedestrians = 0
        num_cars = 0

        with open(filepath, "r") as f:
            lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) == 0:
                    continue

                class_id = int(float(parts[0]))

                if class_id == 0:
                    num_pedestrians += 1
                elif class_id == 2:
                    num_cars += 1

        distribution[(num_pedestrians, num_cars)] += 1

    return distribution


def plot_heatmap(distribution, split_name, output_dir):
    """
    Plot and save heatmap from distribution dictionary.
    """
    if not distribution:
        print(f"No data found for {split_name}")
        return

    max_peds = max(k[0] for k in distribution.keys())
    max_cars = max(k[1] for k in distribution.keys())

    heatmap = np.zeros((max_cars + 1, max_peds + 1))

    for (peds, cars), count in distribution.items():
        heatmap[cars, peds] = count

    split_name = split_name.upper()

    plt.figure(figsize=(4, 3))
    plt.imshow(heatmap, origin="lower", aspect="auto")
    plt.colorbar(label="Number of images")

    plt.xlabel("Number of Pedestrians")
    plt.ylabel("Number of Cars")
    plt.title(f"Object Count Distribution - {split_name}")

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{split_name}_heatmap.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved heatmap to: {save_path}")


def main():
    base_labels_dir = "/ghome/group07/MCV-C5-Group7/ultralytics/dataset/labels/"
    output_dir = "/ghome/group07/MCV-C5-Group7/ultralytics/plots/cooccurence/heatmaps"

    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "val"]:
        labels_dir = os.path.join(base_labels_dir, split)
        distribution = compute_distribution(labels_dir)
        plot_heatmap(distribution, split, output_dir)


if __name__ == "__main__":
    main()
