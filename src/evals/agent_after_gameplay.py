"""After-gameplay propensity eval for a CHECKPOINT agent (not the neutral).

Conditions a fine-tuned checkpoint (e.g. the agreeable or power-seeking agent) on
its OWN game transcript and measures whether its propensities shift vs its turn0
(before-gameplay) baseline. Unlike the qwen3-8b-instruct neutral, these base-SFT
checkpoints are NOT safety-tuned, so the probe can read upward shifts too — making
a null here more credible than the neutral's.

Output mirrors the other evals: results/evals/<persona>_aftergame_<ts>/ with a
2-row matrix (before vs after_treatment) x traits, heatmap, responses JSONL, meta.

    python -m src.evals.agent_after_gameplay \
        --persona agreeable --checkpoint agreeableness/plus \
        --baseline-subject agreeable_ckpt \
        --treatment-dir results/mixed_ft_tinker_ftonly_batch_... --n-items 15
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from src.evals.contagion_eval import load_fresh_baseline
from src.evals.phase_b import build_context_prefix, strip_leaked_instructions
from src.evals.plot_matrix import load_matrix, plot
from src.evals.propensity_eval import (
    BASE_SYSTEM_PROMPT,
    load_items,
    openrouter_client,
    run_propensity_eval,
    tinker_subject,
)
from src.evals.run_eval_matrix import write_table_csv


def extract_persona_transcript(run_path: str | Path, persona: str) -> str:
    """The given persona's final game observation, with leaked instructions stripped."""
    events = [json.loads(l) for l in Path(run_path).read_text().splitlines()]
    run_start = next(e for e in events if e["event"] == "run_start")
    ids = {a["player_id"] for a in run_start["agents"] if a["persona"] == persona}
    acts = [e for e in events if e.get("event") == "agent_action" and e["player_id"] in ids]
    if not acts:
        raise ValueError(f"no '{persona}' agent turns in {run_path}")
    return strip_leaked_instructions(acts[-1]["observation"])


def conditioned_over_persona(
    subject: Any, judge: Any, items: list[dict[str, Any]], runs: list[Path],
    persona: str, common: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_run, records = [], []
    for run_path in runs:
        prefix = build_context_prefix(extract_persona_transcript(run_path, persona))
        summary = run_propensity_eval(subject, BASE_SYSTEM_PROMPT, items, judge,
                                      context_prefix=prefix, **common)
        per_run.append({"run": run_path.name, "mean_score": summary["mean_score"],
                        "n_scored": summary["n_scored"], "n_null": summary["n_null"]})
        for r in summary["rows"]:
            records.append({"run": run_path.name, "item_id": r["id"],
                            "question": r["question"], "response": r["response"], "score": r["score"]})
    scores = [r["mean_score"] for r in per_run if r["mean_score"] is not None]
    return (
        {"mean": statistics.mean(scores) if scores else None,
         "std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
         "n_runs": len(scores), "per_run": per_run},
        records,
    )


def main() -> None:
    from src.runs.run_textarena_game import load_dotenv

    parser = argparse.ArgumentParser(description="After-gameplay propensity eval for a checkpoint agent.")
    parser.add_argument("--persona", required=True, help="Agent persona to condition (e.g. agreeable).")
    parser.add_argument("--checkpoint", required=True, help="Tinker checkpoint ref (e.g. agreeableness/plus).")
    parser.add_argument("--treatment-dir", required=True, help="Game batch to pull this agent's transcripts from.")
    parser.add_argument("--baseline-subject", required=True, help="turn0 matrix row for the before baseline (e.g. agreeable_ckpt).")
    parser.add_argument("--baseline-dir", default="results/evals/turn0_20260608-101059")
    parser.add_argument("--recompute-before", action="store_true",
                        help="Recompute the before baseline on the SAME sampled items (needed with --sample-seed).")
    parser.add_argument("--traits", default="power-seeking,agreeableness")
    parser.add_argument("--judge-model", default="openai/gpt-4o-mini")
    parser.add_argument("--n-items", type=int, default=15)
    parser.add_argument("--sample-seed", type=int, default=0,
                        help="Seed for the random item subsample (same seed across conditions = same items).")
    parser.add_argument("--judge-samples", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    import glob as _glob

    load_dotenv(Path(".env"))
    traits = args.traits.split(",")
    # glob.glob so a wildcard dir (e.g. "results/..._batch_*") spans multiple batches.
    runs = sorted(Path(p) for p in _glob.glob(f"{args.treatment_dir}/run_*.jsonl"))
    if not runs:
        raise SystemExit(f"no run_*.jsonl in {args.treatment_dir}")
    # The turn0 baseline used the FIRST-N items; with a random subsample the before
    # must be recomputed on the SAME items or before->after deltas are confounded.
    before = None if args.recompute_before else load_fresh_baseline(
        args.baseline_dir, traits, subject=args.baseline_subject)

    subject = tinker_subject(args.checkpoint, args.temperature, args.max_tokens)
    judge = openrouter_client(args.judge_model, 1.0, 50)
    common = dict(samples_per_item=1, judge_samples=args.judge_samples, max_workers=args.workers)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir or f"results/evals/{args.persona}_aftergame_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"After-gameplay eval: {args.persona} ({args.checkpoint}) | {len(runs)} transcripts | traits={traits}")
    before_row, after_row = f"{args.persona}_before", f"{args.persona}_after_treatment"
    table: dict[str, dict[str, float | None]] = {before_row: {}, after_row: {}}
    for trait in traits:
        items = load_items(trait, n_items=args.n_items, sample_seed=args.sample_seed)
        if args.recompute_before:
            fresh = run_propensity_eval(subject, BASE_SYSTEM_PROMPT, items, judge, **common)
            before_val = fresh["mean_score"]
        else:
            before_val = before[trait]
        after, records = conditioned_over_persona(subject, judge, items, runs, args.persona, common)
        table[before_row][trait] = before_val
        table[after_row][trait] = after["mean"]
        b, a = before_val, after["mean"]
        shift = "" if (b is None or a is None) else f"  shift {a - b:+.1f}"
        print(f"  [{trait}] before={'n/a' if b is None else round(b,1)} "
              f"after={'n/a' if a is None else round(a,1)} ± {after['std']:.1f} (n={after['n_runs']}){shift}")
        with (out_dir / f"{trait}_responses.jsonl").open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")

    write_table_csv(table, traits, out_dir / "matrix.csv")
    (out_dir / "meta.json").write_text(json.dumps({
        "persona": args.persona, "checkpoint": args.checkpoint,
        "treatment_dir": str(args.treatment_dir), "baseline_subject": args.baseline_subject,
        "n_items": args.n_items, "traits": traits, "runs": [p.name for p in runs],
    }, indent=2))
    subjects, trait_cols, data = load_matrix(str(out_dir / "matrix.csv"))
    plot(subjects, trait_cols, data, str(out_dir / "heatmap.png"),
         f"{args.persona} agent: propensity before vs after gameplay (items={args.n_items})")
    print(f"\nWrote {out_dir}/matrix.csv + heatmap.png")


if __name__ == "__main__":
    main()
