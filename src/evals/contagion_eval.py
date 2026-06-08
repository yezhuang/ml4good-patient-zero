"""Retroactive contagion eval: did the neutral's propensities shift AFTER playing?

Reads already-saved game batches (no replay needed). For the neutral agent it
computes three propensity rows per trait:

  - neutral_fresh           : no game context (the turn0 baseline)
  - neutral_after_control   : conditioned on each CONTROL game it played, averaged
  - neutral_after_treatment : conditioned on each TREATMENT game it played, averaged

Each "after" cell is the mean across that batch's runs (std stored alongside), so
the comparison isn't hostage to a single stochastic game. The contagion read is:
does after_treatment's power-seeking rise above after_control and the fresh
baseline? Output is one comparison heatmap + matrix.csv under results/evals/.

    python -m src.evals.contagion_eval \
        --treatment-dir results/mixed_ft_tinker_ftonly_batch_... \
        --control-dir   results/control_ft_tinker_ftonly_batch_... \
        --n-items 15 --workers 16
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

# Default fresh (turn0) baseline to reuse so we don't recompute it.
DEFAULT_BASELINE_DIR = "results/evals/turn0_20260608-101059"

from src.evals.phase_b import build_context_prefix, extract_neutral_transcript
from src.evals.plot_matrix import load_matrix, plot
from src.evals.propensity_eval import (
    BASE_SYSTEM_PROMPT,
    load_items,
    openrouter_client,
    run_propensity_eval,
)
from src.evals.run_eval_matrix import write_table_csv


def gather_runs(batch_dir: str | Path) -> list[Path]:
    runs = sorted(Path(batch_dir).glob("run_*.jsonl"))
    if not runs:
        raise ValueError(f"no run_*.jsonl found in {batch_dir}")
    return runs


def load_fresh_baseline(
    baseline_dir: str | Path, traits: list[str], subject: str = "neutral"
) -> dict[str, float | None]:
    """Read the neutral fresh-propensity row from a turn0 matrix.csv (no recompute)."""
    path = Path(baseline_dir) / "matrix.csv"
    rows = list(csv.reader(path.open(encoding="utf-8")))
    cols = rows[0][1:]
    row = next((r for r in rows[1:] if r[0] == subject), None)
    if row is None:
        raise ValueError(f"subject '{subject}' not found in {path}")
    by_trait = {c: (float(v) if v != "" else None) for c, v in zip(cols, row[1:])}
    missing = [t for t in traits if t not in by_trait]
    if missing:
        raise ValueError(f"baseline {path} missing traits {missing} (has {cols})")
    return {t: by_trait[t] for t in traits}


def conditioned_over_batch(
    subject: Any, judge: Any, items: list[dict[str, Any]], runs: list[Path], common: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the conditioned eval once per game in a batch; aggregate per-run means.

    Returns (aggregate, records) where `records` is the full per-(run, item)
    question/response/score log for auditing — written to a responses JSONL.
    """
    per_run = []
    records: list[dict[str, Any]] = []
    for run_path in runs:
        transcript, _ = extract_neutral_transcript(run_path)
        prefix = build_context_prefix(transcript)
        summary = run_propensity_eval(subject, BASE_SYSTEM_PROMPT, items, judge,
                                      context_prefix=prefix, **common)
        per_run.append({"run": run_path.name, "mean_score": summary["mean_score"],
                        "n_scored": summary["n_scored"]})
        for r in summary["rows"]:
            records.append({"run": run_path.name, "item_id": r["id"],
                            "question": r["question"], "response": r["response"],
                            "score": r["score"]})
    scores = [r["mean_score"] for r in per_run if r["mean_score"] is not None]
    aggregate = {
        "mean": statistics.mean(scores) if scores else None,
        "std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "n_runs": len(scores),
        "per_run": per_run,
    }
    return aggregate, records


def main() -> None:
    from src.runs.run_textarena_game import load_dotenv

    parser = argparse.ArgumentParser(description="Retroactive contagion eval over saved batches.")
    parser.add_argument("--treatment-dir", required=True, help="Treatment batch dir (mixed).")
    parser.add_argument("--control-dir", required=True, help="Control batch dir.")
    parser.add_argument(
        "--baseline-dir", default=DEFAULT_BASELINE_DIR,
        help="turn0 matrix dir to reuse the neutral fresh row from (avoids recompute). "
        "Pass 'recompute' to compute fresh from scratch instead.",
    )
    parser.add_argument("--traits", default="power-seeking,agreeableness")
    parser.add_argument("--subject-model", default=None, help="Override neutral model id.")
    parser.add_argument("--judge-model", default="openai/gpt-4o-mini")
    parser.add_argument("--n-items", type=int, default=15)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--judge-samples", type=int, default=3)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    treatment_runs = gather_runs(args.treatment_dir)
    control_runs = gather_runs(args.control_dir)
    _, neutral_model = extract_neutral_transcript(treatment_runs[0])
    model = args.subject_model or (
        neutral_model if neutral_model and not str(neutral_model).startswith("tinker://")
        else "qwen/qwen3-8b"
    )
    traits = args.traits.split(",")

    subject = openrouter_client(model, 1.0, args.max_tokens)
    judge = openrouter_client(args.judge_model, 1.0, 50)
    common = dict(samples_per_item=args.samples, judge_samples=args.judge_samples,
                  max_workers=args.workers)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir or f"results/evals/contagion_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    recompute_fresh = str(args.baseline_dir).lower() == "recompute"
    baseline = None if recompute_fresh else load_fresh_baseline(args.baseline_dir, traits)
    fresh_src = "recomputed" if recompute_fresh else f"reused from {args.baseline_dir}"
    print(f"Contagion eval: neutral={model} | treatment={len(treatment_runs)} runs, "
          f"control={len(control_runs)} runs | traits={traits}")
    print(f"  neutral_fresh: {fresh_src}")
    rows = ["neutral_fresh", "neutral_after_control", "neutral_after_treatment"]
    table: dict[str, dict[str, float | None]] = {r: {} for r in rows}
    detail: dict[str, Any] = {}
    for trait in traits:
        items = load_items(trait, n_items=args.n_items)
        if recompute_fresh:
            fresh_summary = run_propensity_eval(subject, BASE_SYSTEM_PROMPT, items, judge, **common)
            fresh_score = fresh_summary["mean_score"]
        else:
            fresh_score = baseline[trait]
            fresh_summary = {"mean_score": fresh_score, "source": str(args.baseline_dir)}
        ctrl, ctrl_recs = conditioned_over_batch(subject, judge, items, control_runs, common)
        treat, treat_recs = conditioned_over_batch(subject, judge, items, treatment_runs, common)
        table["neutral_fresh"][trait] = fresh_score
        table["neutral_after_control"][trait] = ctrl["mean"]
        table["neutral_after_treatment"][trait] = treat["mean"]
        detail[trait] = {"fresh": fresh_summary, "after_control": ctrl, "after_treatment": treat}

        # Full question + response + score log, one line per (condition, run, item).
        with (out_dir / f"{trait}_responses.jsonl").open("w", encoding="utf-8") as fh:
            for rec in ctrl_recs:
                fh.write(json.dumps({"condition": "after_control", **rec}) + "\n")
            for rec in treat_recs:
                fh.write(json.dumps({"condition": "after_treatment", **rec}) + "\n")

        def fmt(v: float | None) -> str:
            return "n/a" if v is None else f"{v:.1f}"
        print(f"\n  [{trait}]")
        print(f"    fresh            = {fmt(fresh_score)}")
        print(f"    after_control    = {fmt(ctrl['mean'])} ± {ctrl['std']:.1f}  (n={ctrl['n_runs']})")
        print(f"    after_treatment  = {fmt(treat['mean'])} ± {treat['std']:.1f}  (n={treat['n_runs']})")
        if treat["mean"] is not None and ctrl["mean"] is not None:
            print(f"    treatment - control = {treat['mean'] - ctrl['mean']:+.1f}")
        (out_dir / f"{trait}.json").write_text(json.dumps(detail[trait], indent=2))

    write_table_csv(table, traits, out_dir / "matrix.csv")
    (out_dir / "meta.json").write_text(json.dumps({
        "stage": "contagion", "note": "fresh vs after_control vs after_treatment (neutral)",
        "fresh_source": "recomputed" if recompute_fresh else str(args.baseline_dir),
        "treatment_dir": str(args.treatment_dir), "control_dir": str(args.control_dir),
        "treatment_runs": [p.name for p in treatment_runs],
        "control_runs": [p.name for p in control_runs],
        "n_items": args.n_items, "traits": traits, "subject": model,
    }, indent=2))
    subjects, trait_cols, data = load_matrix(str(out_dir / "matrix.csv"))
    plot(subjects, trait_cols, data, str(out_dir / "heatmap.png"),
         f"Contagion: neutral propensity by gameplay condition (n={args.n_items})")
    print(f"\nWrote {out_dir}/matrix.csv + heatmap.png")


if __name__ == "__main__":
    main()
