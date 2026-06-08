"""Bar chart of the NEUTRAL agent's defect rate (the contagion rate) per condition.

Each condition is a game batch; the bar is the neutral's mean valid-only defect
rate across runs (error bar = std). Reference conditions (control/baseline) show
the no-bad-agent floor, so contagion = how far a bad-agent bar rises above it.

    python -m src.analysis.contagion_bar \
        --bad "power-seeking=results/mixed_ft_tinker_ftonly_batch_..." \
        --bad "always-defect=results/mixed_alwaysdefect_ftonly_batch_..." \
        --ref "control=results/control_ft_tinker_ftonly_batch_..." \
        --out results/contagion_bar.png
"""

from __future__ import annotations

import argparse
import glob
import statistics
from pathlib import Path

from src.analysis.analyze_runs import analyze_run
from src.analysis.compare_conditions import per_round_rows


def neutral_defect(batch_dir: str) -> tuple[float, float, int]:
    """Mean, std, n of the neutral's valid-only defect rate across a batch's runs."""
    rates: list[float] = []
    for rf in sorted(glob.glob(f"{batch_dir}/run_*.jsonl")):
        summary = analyze_run(Path(rf))
        per_player: dict[str, list[float]] = {}
        for row in per_round_rows(summary):
            if row["persona"] != "neutral":
                continue
            per_player.setdefault(row["player_id"], []).append(row["defect_rate"])
        for player_rates in per_player.values():
            if player_rates:
                rates.append(statistics.mean(player_rates))
    if not rates:
        return float("nan"), 0.0, 0
    return statistics.mean(rates), (statistics.pstdev(rates) if len(rates) > 1 else 0.0), len(rates)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--bad", action="append", default=[], help="label=batch_dir (bad-agent condition).")
    parser.add_argument("--ref", action="append", default=[], help="label=batch_dir (reference: control/baseline).")
    parser.add_argument("--out", default="results/contagion_bar.png")
    parser.add_argument("--title", default="Contagion: neutral defect rate by condition")
    args = parser.parse_args()

    labels, means, cis, colors = [], [], [], []
    for spec, is_bad in [(s, True) for s in args.bad] + [(s, False) for s in args.ref]:
        label, batch = spec.split("=", 1)
        m, sd, n = neutral_defect(batch)
        # 95% CI of the mean (the uncertainty of the per-game average), not ±1 SD
        # (the raw spread). Normal approx; fine for the n we report.
        ci = 1.96 * sd / (n ** 0.5) if n > 1 else 0.0
        labels.append(label)
        means.append(m)
        cis.append(ci)
        colors.append("#c0392b" if is_bad else "#95a5a6")  # red = bad agent, grey = reference
        print(f"  {label:18s} neutral defect = {m:.2f}  95% CI ±{ci:.2f}  (SD {sd:.2f}, n={n})")

    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(labels), 4.2))
    bars = ax.bar(labels, means, yerr=cis, color=colors, capsize=5, edgecolor="black", linewidth=0.6)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.02, f"{m:.2f}", ha="center", fontweight="bold")
    ax.set_ylabel("neutral defect rate (contagion)\n(error bars = 95% CI of the mean)")
    ax.set_ylim(0, 1.0)
    ax.set_title(args.title, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#c0392b"), plt.Rectangle((0, 0), 1, 1, color="#95a5a6")]
    ax.legend(handles, ["bad agent present", "reference (no bad agent)"], fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
