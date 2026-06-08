"""Plot defect rate per round by condition from a comparison tidy CSV.

Reads the `*_tidy.csv` produced by `compare_conditions` and draws one line per
condition (mean defect rate per round) with a ±std band across runs/agents — so
single runs show flat lines and batches show error bands. Defaults to the neutral
role, which is the contagion target.

    python -m src.analysis.plot_contagion --tidy results/contagion_tidy.csv
    python -m src.analysis.plot_contagion --persona all   # overlay every role
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write a file, no display needed
import matplotlib.pyplot as plt  # noqa: E402


def runs_per_condition(rows: list[dict]) -> dict[str, int]:
    """Number of source logs per condition.

    New comparison CSVs include source provenance because run_id is the config
    name and may repeat across batch rows. Fall back to the older inference for
    legacy tidy files that do not have source_run_index/source_path columns.
    """
    source_ids: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        source_id = row.get("source_path") or row.get("source_run_index")
        if source_id not in (None, ""):
            source_ids[row["condition"]].add(str(source_id))
    if source_ids:
        return {condition: len(ids) for condition, ids in source_ids.items()}

    counts = Counter((r["condition"], r["player_id"], r["round"]) for r in rows)
    runs: dict[str, int] = defaultdict(int)
    for (condition, _, _), count in counts.items():
        runs[condition] = max(runs[condition], count)
    return dict(runs)


def sample_size_label(rows: list[dict]) -> str:
    """A short tag like 'n1' (uniform) or 'n1-10' (varying) for filenames/titles."""
    values = set(runs_per_condition(rows).values())
    if len(values) == 1:
        return f"n{values.pop()}"
    return f"n{min(values)}-{max(values)}"


def load_tidy(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["round"] = int(row["round"])
            row["defect_rate"] = float(row["defect_rate"])
            rows.append(row)
    return rows


def plot(
    rows: list[dict],
    persona: str,
    out_path: str,
    title: str,
    ylabel: str = "defect rate",
    band: bool = True,
) -> str:
    # condition -> round -> [defect rates across runs and matching agents]
    data: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if persona != "all" and row["persona"] != persona:
            continue
        data[row["condition"]][row["round"]].append(row["defect_rate"])

    if not data:
        raise SystemExit(f"No rows for persona={persona!r} in the tidy CSV.")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for condition in sorted(data):
        rounds = sorted(data[condition])
        means = [statistics.mean(data[condition][r]) for r in rounds]
        ax.plot(rounds, means, marker="o", linewidth=2, label=condition)
        # The ±std band is informative for a couple of conditions but turns into
        # unreadable overlapping fill with many lines, so it's optional.
        if band:
            stds = [
                statistics.pstdev(data[condition][r]) if len(data[condition][r]) > 1 else 0.0
                for r in rounds
            ]
            lower = [max(0.0, m - s) for m, s in zip(means, stds)]
            upper = [min(1.0, m + s) for m, s in zip(means, stds)]
            ax.fill_between(rounds, lower, upper, alpha=0.15)

    all_rounds = sorted({row["round"] for row in rows})
    ax.set_xticks(all_rounds)
    ax.set_xlabel("Round")
    ax.set_ylabel(f"{persona} {ylabel}")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.legend(title="condition")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tidy", default="results/contagion_tidy.csv")
    parser.add_argument("--persona", default="neutral", help="Role to plot, or 'all'.")
    parser.add_argument("--out", default="results/contagion_plot.png")
    parser.add_argument(
        "--title", default="Misalignment contagion: defect rate by round"
    )
    parser.add_argument(
        "--ylabel", default="defect rate", help="Y-axis metric name (e.g. 'free-ride rate')."
    )
    parser.add_argument(
        "--no-band", action="store_true", help="Hide the ±std shaded band (cleaner with many lines)."
    )
    args = parser.parse_args()

    rows = load_tidy(args.tidy)
    n_label = sample_size_label(rows)

    # Embed the sample size in the title and filename so plots are never
    # mistaken for higher-n results.
    title = f"{args.title} ({n_label})"
    out = Path(args.out)
    out = str(out.with_name(f"{out.stem}_{n_label}{out.suffix}"))

    out = plot(rows, args.persona, out, title, ylabel=args.ylabel, band=not args.no_band)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
