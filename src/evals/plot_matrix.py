"""Render a propensity-eval matrix CSV as an annotated heatmap (subjects x traits)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load_matrix(csv_path: str) -> tuple[list[str], list[str], np.ndarray]:
    rows = list(csv.reader(open(csv_path, encoding="utf-8")))
    traits = rows[0][1:]
    subjects, data = [], []
    for row in rows[1:]:
        subjects.append(row[0])
        data.append([float(x) if x != "" else np.nan for x in row[1:]])
    return subjects, traits, np.array(data)


def plot(
    subjects,
    traits,
    data,
    out_path: str,
    title: str,
    *,
    xlabel: str = "eval (trait)",
    ylabel: str = "subject (agent)",
    cbar_label: str = "propensity score (0-100)",
) -> str:
    fig, ax = plt.subplots(figsize=(2.0 + 1.7 * len(traits), 1.4 + 0.8 * len(subjects)))
    im = ax.imshow(data, cmap="RdYlBu_r", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(traits)))
    ax.set_xticklabels(traits, rotation=20, ha="right")
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(subjects)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    for i in range(len(subjects)):
        for j in range(len(traits)):
            v = data[i, j]
            if not np.isnan(v):
                # white text on dark (extreme) cells, black otherwise
                color = "white" if (v < 25 or v > 75) else "black"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color=color, fontsize=13, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Path to a matrix.csv.")
    parser.add_argument("--matrix-dir", help="A matrix_<ts>/ dir (uses its matrix.csv).")
    parser.add_argument("--out", default=None)
    parser.add_argument("--title", default="Propensity eval: agents x traits")
    args = parser.parse_args()

    csv_path = args.csv or str(Path(args.matrix_dir) / "matrix.csv")
    out = args.out or str(Path(csv_path).with_name("heatmap.png"))
    subjects, traits, data = load_matrix(csv_path)
    plot(subjects, traits, data, out, args.title)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
