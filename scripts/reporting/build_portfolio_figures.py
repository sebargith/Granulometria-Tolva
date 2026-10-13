"""Render shareable charts from aggregate counts and synthetic labels only."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]


def main():
    target = ROOT / "docs/public"
    data = json.loads((target / "results.json").read_text(encoding="utf-8"))
    matrix = data["experiments"][-1]["matrix"]
    fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=30)
    ax.set(xticks=range(3), yticks=range(3),
           xticklabels=["Fine", "Medium", "Coarse"],
           yticklabels=["Fine", "Medium", "Coarse"],
           xlabel="Predicted dominant class", ylabel="Visual reference class")
    ax.set_title("Full pipeline | 90-frame technical test", pad=16)
    for row in range(3):
        for col in range(3):
            ax.text(col, row, str(matrix[row][col]), ha="center", va="center",
                    fontsize=18, color="white" if matrix[row][col] > 15 else "#202020")
    fig.savefig(target / "technical_confusion.png", dpi=180)
    plt.close(fig)

    fixture = json.loads((ROOT / "examples/synthetic_frames.json").read_text(encoding="utf-8"))
    labels = fixture["frames"][0]["labels"]
    colors = {"fino": "#309345", "medio": "#f0cf43", "grueso": "#d65343"}
    fig, ax = plt.subplots(figsize=(8, 3.2), layout="constrained")
    for i, label in enumerate(labels):
        x, y = i % 6, 1 - i // 6
        ax.add_patch(Rectangle((x, y), .95, .95, facecolor=colors[label]))
        ax.text(x+.475, y+.475, f"{i+1:02d}\n{label}", ha="center", va="center", fontsize=11)
    ax.set(xlim=(-.1, 6), ylim=(-.1, 2.1), aspect="equal")
    ax.axis("off")
    ax.set_title("Synthetic patch labels | NOT a model prediction", pad=15)
    fig.savefig(target / "synthetic_patches.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
