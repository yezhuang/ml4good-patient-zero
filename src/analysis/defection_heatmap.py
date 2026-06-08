"""Heatmap of in-game defect rate by role, across game-batch conditions.

Rows = conditions (e.g. which bad agent), columns = role (bad / agreeable /
neutral). Each cell is the mean valid-only defect rate across that batch's runs.
Lets you read both the bad agent's strength AND the spillover to the others.

    python -m src.analysis.defection_heatmap \
        --condition "power-seeking=results/mixed_ft_tinker_ftonly_batch_..." \
        --condition "spiteful=results/mixed_spiteful_ftonly_batch_..." \
        --condition "cooperation-minus=results/mixed_coopminus_ftonly_batch_..." \
        --out results/defection_heatmap.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import statistics
from pathlib import Path

from src.analysis.analyze_runs import analyze_run
from src.analysis.compare_conditions import per_round_rows

ROLES = ["bad_agent", "agreeable", "neutral"]


def _role(persona: str) -> str:
    if persona == "neutral":
        return "neutral"
    if persona == "agreeable":
        return "agreeable"
    return "bad_agent"


def role_defect_rates(batch_dir: str) -> dict[str, float | None]:
    """Mean valid-only defect rate per role across all runs in a batch."""
    per_role: dict[str, list[float]] = {r: [] for r in ROLES}
    for rf in sorted(glob.glob(f"{batch_dir}/run_*.jsonl")):
        summary = analyze_run(Path(rf))
        per_player: dict[str, tuple[str, list[float]]] = {}
        for row in per_round_rows(summary):
            persona, rates = per_player.setdefault(row["player_id"], (row["persona"], []))
            rates.append(row["defect_rate"])
        for persona, rates in per_player.values():
            if rates:
                per_role[_role(persona)].append(statistics.mean(rates))
    return {r: (statistics.mean(v) if v else None) for r, v in per_role.items()}


def main() -> None:
    from src.evals.plot_matrix import plot

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition", action="append", required=True,
        help="label=batch_dir (repeatable, in row order).",
    )
    parser.add_argument("--out", default="results/defection_heatmap.png")
    parser.add_argument("--title", default="In-game defect rate by role")
    args = parser.parse_args()

    table: dict[str, dict[str, float | None]] = {}
    for spec in args.condition:
        label, batch_dir = spec.split("=", 1)
        table[label] = role_defect_rates(batch_dir)
        cells = " ".join(
            f"{r}={'n/a' if table[label][r] is None else f'{table[label][r]:.2f}'}" for r in ROLES
        )
        print(f"  {label:18s} {cells}")

    csv_path = Path(args.out).with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["condition"] + ROLES)
        for label, row in table.items():
            w.writerow([label] + ["" if row[r] is None else round(row[r], 2) for r in ROLES])

    # Defect rate is 0-1; scale to 0-100 so it reuses the shared 0-100 heatmap.
    import numpy as np
    data = np.array(
        [[(np.nan if table[c][r] is None else table[c][r] * 100) for r in ROLES] for c in table]
    )
    plot(list(table), ROLES, data, args.out, args.title,
         xlabel="role in game", ylabel="condition (bad agent)",
         cbar_label="defect rate (%)")
    print(f"\nWrote {args.out} and {csv_path}")


if __name__ == "__main__":
    main()
