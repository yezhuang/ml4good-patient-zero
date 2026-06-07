"""Run a config N times and aggregate the neutral agents' defection rates.

LLM games are stochastic, so a single run proves nothing. This runs the same
config `--runs` times and reports the neutral agents' mean (± std) defect rate,
per round and overall. Compare two batches (e.g. treatment vs control) to read
off a contagion effect:

    python -m src.runs.run_batch --config configs/mixed_ft_tinker.json   --runs 10
    python -m src.runs.run_batch --config configs/control_ft_tinker.json --runs 10

Each batch writes its per-run logs and an aggregate.json under one timestamped
results/<run_id>_batch_<ts>/ directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from src.analysis.analyze_runs import analyze_run
from src.runs.run_textarena_game import load_dotenv, run_textarena_game

logger = logging.getLogger("ml4good.batch")


def _round_defect_rate(round_entry: dict[str, Any]) -> float:
    applied = round_entry["applied"].values()
    if not applied:
        return 0.0
    return sum(1 for action in applied if action == "defect") / len(applied)


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "mean": statistics.mean(values) if values else None,
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
        "values": values,
    }


def aggregate_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate neutral defect rates across per-run analyzer summaries.

    Takes the output of `analyze_run` for each run (must contain an `ipd` block).
    """
    overall: dict[str, list[float]] = defaultdict(list)
    per_round: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    labels: dict[str, str] = {}
    runs_used = 0

    for summary in summaries:
        ipd = summary.get("ipd")
        if not ipd:
            continue
        runs_used += 1
        for pid in ipd["neutral_players"]:
            st = ipd["players"][pid]
            labels[pid] = st["label"]
            total = st["applied_defect"] + st["applied_cooperate"]
            overall[pid].append(st["applied_defect"] / total if total else 0.0)
            for entry in st["rounds"]:
                per_round[pid][entry["round"]].append(_round_defect_rate(entry))

    neutral_players = {
        pid: {
            "label": labels[pid],
            "overall_defect_rate": _stats(overall[pid]),
            "per_round_defect_rate": {
                str(rnd): _stats(per_round[pid][rnd])
                for rnd in sorted(per_round[pid])
            },
        }
        for pid in sorted(overall)
    }
    return {"runs": runs_used, "neutral_players": neutral_players}


def run_batch(config: dict[str, Any], runs: int, out_dir: str | None = None) -> Path:
    run_id = config["run_id"]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_dir = Path(out_dir or f"results/{run_id}_batch_{stamp}")
    batch_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for i in range(1, runs + 1):
        run_config = config_for_batch_run(config, i, batch_dir)
        try:
            path = run_textarena_game(run_config)
            summaries.append(analyze_run(path))
            logger.info("[%d/%d] ok: %s", i, runs, path)
        except Exception as exc:  # one bad game shouldn't kill the whole batch
            logger.exception("[%d/%d] run failed: %s", i, runs, exc)
            failures.append({"run_index": i, "error": repr(exc)})

    if failures:
        logger.warning("%d/%d runs failed; aggregating the %d that succeeded",
                       len(failures), runs, len(summaries))

    aggregate = aggregate_summaries(summaries)
    aggregate["run_id"] = run_id
    aggregate["requested_runs"] = runs
    aggregate["failed_runs"] = failures
    (batch_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(format_batch_summary(aggregate))
    print(f"\nWrote {batch_dir}/aggregate.json")
    return batch_dir


def config_for_batch_run(
    config: dict[str, Any], run_index: int, batch_dir: Path
) -> dict[str, Any]:
    """Prepare one run config, varying seed when player randomization is enabled."""
    run_config = dict(config)
    run_config["output_path"] = str(batch_dir / f"run_{run_index:02d}.jsonl")
    run_config["timestamp_output"] = False
    if run_config.get("randomize_player_ids", False):
        run_config["seed"] = int(config.get("seed", 0)) + run_index - 1
    return run_config


def format_batch_summary(aggregate: dict[str, Any]) -> str:
    lines = [
        "",
        f"Batch: {aggregate.get('run_id')} | runs aggregated: {aggregate['runs']}",
        "Neutral agents' defect rate (mean ± std across runs):",
    ]
    for pid, data in aggregate["neutral_players"].items():
        o = data["overall_defect_rate"]
        mean = f"{o['mean']:.2f}" if o["mean"] is not None else "n/a"
        lines.append(f"- P{pid} {data['label']}: overall {mean} ± {o['std']:.2f}")
        per_round = " ".join(
            f"R{rnd}={s['mean']:.2f}±{s['std']:.2f}"
            for rnd, s in data["per_round_defect_rate"].items()
        )
        if per_round:
            lines.append(f"    by round: {per_round}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to run config JSON.")
    parser.add_argument("--runs", type=int, default=10, help="Number of games to run.")
    parser.add_argument("--out-dir", help="Override the batch output directory.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv(Path(".env"))
    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)
    run_batch(config, runs=args.runs, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
