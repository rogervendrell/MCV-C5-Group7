from pathlib import Path
from datetime import datetime


def make_summary(metrics):
    lines = []

    def add(line=""):
        """Append a line safely as string."""
        lines.append(str(line))

    # Header
    add("YOLO Validation Summary")
    add("========================")
    # add(f"Dataset : {args.dataset}")
    # add(f"Load VP : {args.load_vp}")
    # add(f"Fusion  : {args.fusion}")
    # add(f"Task    : {metrics.task}")
    # add()

    # Key metrics
    add("Key Metrics (results_dict)")
    add("--------------------------")
    for k, v in metrics.results_dict.items():
        if isinstance(v, float):
            add(f"{k:30s}: {v:.6f}")
        else:
            add(f"{k:30s}: {v}")
    add()

    # Ultralytics summary
    add("Ultralytics Summary")
    add("-------------------")
    for line in metrics.summary():
        add(line)
    add()

    # Speed
    add("Speed (ms/img)")
    add("--------------")
    for k, v in metrics.speed.items():
        add(f"{k:12s}: {v:.2f}")

    return "\n".join(lines)