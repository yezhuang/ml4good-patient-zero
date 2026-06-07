"""Combine runs from multiple conditions into one plot-ready summary.

Point it at the run logs for each experimental condition (single runs or whole
batches via globs). It emits:

  <out>_tidy.csv      long format, one row per (condition, run, player, round) —
                      drop straight into pandas/seaborn for trend plots with CIs
  <out>_summary.json  per-condition aggregates by role (persona): mean/std/n
                      defect rate per round and overall

Aggregation is keyed by persona (role), not player_id, so it is robust to
player-id randomization and to conditions with several agents of one role.

    python -m src.analysis.compare_conditions \
        --condition "baseline=results/baseline_neutral_*.jsonl" \
        --condition "control=results/control_ft_tinker_batch_*/run_*.jsonl" \
        --condition "treatment=results/mixed_ft_tinker_batch_*/run_*.jsonl" \
        --out results/comparison
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.analysis.analyze_runs import analyze_run


def per_round_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-player, per-round defect rate from one analyzer summary."""
    ipd = summary.get("ipd")
    if not ipd:
        return []
    rows = []
    for pid, st in ipd["players"].items():
        for entry in st["rounds"]:
            applied = list(entry["applied"].values())
            rate = (
                sum(1 for a in applied if a == "defect") / len(applied)
                if applied
                else 0.0
            )
            rows.append(
                {
                    "player_id": pid,
                    "label": st["label"],
                    "persona": st["persona"],
                    "round": entry["round"],
                    "defect_rate": rate,
                }
            )
    return rows


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "mean": statistics.mean(values) if values else None,
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def build_comparison(
    condition_to_summaries: dict[str, list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (tidy_rows, summary) across all conditions."""
    tidy: list[dict[str, Any]] = []
    conditions: dict[str, Any] = {}

    for condition, summaries in condition_to_summaries.items():
        # per (persona, round) -> rates; and per (persona) -> per-run overall rates
        by_persona_round: dict[str, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        by_persona_overall: dict[str, list[float]] = defaultdict(list)
        runs_used = 0

        for summary in summaries:
            rows = per_round_rows(summary)
            if not rows:
                continue
            runs_used += 1
            run_id = summary.get("run_id", "")
            # per-player overall (mean of its rounds) for this run
            per_player_rates: dict[str, list[float]] = defaultdict(list)
            per_player_persona: dict[str, str] = {}
            for row in rows:
                tidy.append({"condition": condition, "run_id": run_id, **row})
                by_persona_round[row["persona"]][row["round"]].append(row["defect_rate"])
                per_player_rates[row["player_id"]].append(row["defect_rate"])
                per_player_persona[row["player_id"]] = row["persona"]
            for pid, rates in per_player_rates.items():
                by_persona_overall[per_player_persona[pid]].append(
                    statistics.mean(rates)
                )

        conditions[condition] = {
            "runs": runs_used,
            "by_persona": {
                persona: {
                    "overall_defect_rate": _stats(by_persona_overall[persona]),
                    "per_round_defect_rate": {
                        str(rnd): _stats(by_persona_round[persona][rnd])
                        for rnd in sorted(by_persona_round[persona])
                    },
                }
                for persona in sorted(by_persona_round)
            },
        }

    return tidy, {"conditions": conditions}


def write_outputs(
    tidy: list[dict[str, Any]], summary: dict[str, Any], out_prefix: str
) -> tuple[Path, Path]:
    out = Path(out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out.with_name(out.name + "_tidy.csv")
    json_path = out.with_name(out.name + "_summary.json")

    fields = ["condition", "run_id", "player_id", "label", "persona", "round", "defect_rate"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(tidy)

    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return csv_path, json_path


def format_comparison(summary: dict[str, Any]) -> str:
    lines = ["", "Condition comparison — defect rate by role (mean ± std):"]
    for condition, data in summary["conditions"].items():
        lines.append(f"  [{condition}]  runs={data['runs']}")
        for persona, pdata in data["by_persona"].items():
            o = pdata["overall_defect_rate"]
            mean = f"{o['mean']:.2f}" if o["mean"] is not None else "n/a"
            per_round = " ".join(
                f"R{rnd}={s['mean']:.2f}"
                for rnd, s in pdata["per_round_defect_rate"].items()
            )
            lines.append(f"    {persona:14s} overall {mean}±{o['std']:.2f}  | {per_round}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        metavar="NAME=GLOB",
        help="Condition name and a glob of its run .jsonl files. Repeatable.",
    )
    parser.add_argument("--out", default="results/comparison", help="Output path prefix.")
    args = parser.parse_args()

    condition_to_summaries: dict[str, list[dict[str, Any]]] = {}
    for spec in args.condition:
        name, _, pattern = spec.partition("=")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"warning: no files matched for {name!r}: {pattern}")
        condition_to_summaries[name] = [analyze_run(Path(f)) for f in files]

    tidy, summary = build_comparison(condition_to_summaries)
    csv_path, json_path = write_outputs(tidy, summary, args.out)
    print(format_comparison(summary))
    print(f"\nWrote {csv_path}\nWrote {json_path}")


if __name__ == "__main__":
    main()
