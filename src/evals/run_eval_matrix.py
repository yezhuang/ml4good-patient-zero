"""Run a subject x trait propensity-eval matrix and tabulate the mean scores.

Phase A of the contagion eval: characterize each agent's BASELINE propensities
(no game context). Subjects are sampled via this repo's clients (a Tinker
checkpoint is loaded once per subject); each (subject, trait) cell is judged with
gpt-4o-mini and aggregated to a mean trait score.

    python -m src.evals.run_eval_matrix --n-items 20
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.evals.propensity_eval import (
    BASE_SYSTEM_PROMPT,
    load_items,
    openrouter_client,
    run_propensity_eval,
    tinker_subject,
)

# (subject label) -> (kind, value): "checkpoint" = Tinker ref, "model" = OpenRouter id.
DEFAULT_SUBJECTS: dict[str, tuple[str, str]] = {
    "power_seeking_ckpt": ("checkpoint", "power-seeking/plus"),
    "agreeable_ckpt": ("checkpoint", "agreeableness/plus"),
    "neutral": ("model", "qwen/qwen3-8b"),
}
# Just the two traits that match the game's agents (and the only two we track).
DEFAULT_TRAITS = ["power-seeking", "agreeableness"]


def build_subject(kind: str, value: str, temperature: float, max_tokens: int) -> Any:
    if kind == "checkpoint":
        return tinker_subject(value, temperature, max_tokens)
    return openrouter_client(value, temperature, max_tokens)


def run_matrix(
    subjects: dict[str, tuple[str, str]],
    traits: list[str],
    judge: Any,
    *,
    n_items: int | None,
    system_prompt: str,
    samples_per_item: int,
    paraphrases: int,
    judge_samples: int,
    temperature: float,
    max_tokens: int,
    out_dir: Path,
    max_workers: int = 8,
) -> dict[str, dict[str, float | None]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    table: dict[str, dict[str, float | None]] = {}
    for sname, (kind, value) in subjects.items():
        print(f"\n=== subject {sname} ({value}) ===", flush=True)
        subject = build_subject(kind, value, temperature, max_tokens)
        table[sname] = {}
        for trait in traits:
            items = load_items(trait, n_items=n_items)
            summary = run_propensity_eval(
                subject, system_prompt, items, judge,
                paraphrases_per_item=paraphrases, samples_per_item=samples_per_item,
                judge_samples=judge_samples, max_workers=max_workers,
            )
            mean = summary["mean_score"]
            table[sname][trait] = mean
            mean_s = "n/a" if mean is None else f"{mean:.1f}"
            print(f"  {trait}: {mean_s} (scored {summary['n_scored']}/{summary['n_responses']})", flush=True)
            (out_dir / f"{sname}__{trait}.json").write_text(json.dumps(summary, indent=2))
    return table


def format_table(table: dict[str, dict[str, float | None]], traits: list[str]) -> str:
    width = max((len(s) for s in table), default=8) + 2
    header = "subject".ljust(width) + "".join(t[:13].rjust(14) for t in traits)
    lines = [header]
    for sname, row in table.items():
        cells = "".join(
            ("n/a" if row.get(t) is None else f"{row[t]:.1f}").rjust(14) for t in traits
        )
        lines.append(sname.ljust(width) + cells)
    return "\n".join(lines)


def write_table_csv(table: dict[str, dict[str, float | None]], traits: list[str], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject"] + traits)
        for sname, row in table.items():
            writer.writerow([sname] + ["" if row.get(t) is None else round(row[t], 1) for t in traits])


def dry_run(
    subjects: dict[str, tuple[str, str]],
    traits: list[str],
    n_items: int,
    system_prompt: str,
    judge_samples: int,
) -> None:
    """Validate the matrix wiring (clients resolve, items load, prompts build) — no calls."""
    print(f"DRY RUN — matrix: {len(subjects)} subjects x {len(traits)} traits x {n_items} items")
    print("(no model/judge calls; no checkpoint loading)\n")

    print("Subjects:")
    for sname, (kind, value) in subjects.items():
        if kind == "checkpoint":
            from src.models.tinker_registry import resolve_checkpoint
            try:
                uri = resolve_checkpoint(value)
                print(f"  {sname:20s} tinker checkpoint {value} -> {uri}")
            except Exception as exc:
                print(f"  {sname:20s} ERROR resolving {value}: {exc}")
        else:
            print(f"  {sname:20s} openrouter model {value}")

    print("\nTraits:")
    for trait in traits:
        items = load_items(trait, n_items=n_items)
        metric = next(iter(items[0]["judge_prompts"].keys()))
        print(f"  {trait:16s} loaded {len(items)} items (metric={metric})")

    items = load_items(traits[0], n_items=n_items)
    it = items[0]
    scenario = it["paraphrases"][0]
    metric, judge_prompt = next(iter(it["judge_prompts"].items()))
    print(f"\n=== EXAMPLE cell: trait={traits[0]} item={it['id']} ===")
    print("--- subject SYSTEM ---\n" + system_prompt)
    print("\n--- subject USER (eval scenario, head 400) ---\n" + scenario[:400])
    filled = judge_prompt.replace("{question}", "<scenario>").replace(
        "{answer}", "<the subject's response goes here>"
    )
    print("\n--- judge prompt (filled, tail 350) ---\n" + filled[-350:])

    n_resp = len(subjects) * len(traits) * n_items
    print(f"\nWould make ~{n_resp} subject responses + ~{n_resp * judge_samples} judge calls.")


def main() -> None:
    from src.runs.run_textarena_game import load_dotenv

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-items", type=int, default=20, help="Test items per trait.")
    parser.add_argument("--samples", type=int, default=1, help="Subject samples per paraphrase.")
    parser.add_argument("--paraphrases", type=int, default=1)
    parser.add_argument("--judge-samples", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8, help="Concurrent requests (1=sequential).")
    parser.add_argument("--judge-model", default="openai/gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--system-prompt", default="base", choices=["base", "none"])
    parser.add_argument("--traits", default=",".join(DEFAULT_TRAITS))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--stage", default="turn0",
        help="Gameplay stage label for the output dir (turn0 = before gameplay).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Verify clients/items/prompts without any model or judge calls.",
    )
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    traits = args.traits.split(",")
    system_prompt = BASE_SYSTEM_PROMPT if args.system_prompt == "base" else ""

    if args.dry_run:
        dry_run(DEFAULT_SUBJECTS, traits, args.n_items, system_prompt, args.judge_samples)
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir or f"results/evals/{args.stage}_{stamp}")
    judge = openrouter_client(args.judge_model, 1.0, 50)

    table = run_matrix(
        DEFAULT_SUBJECTS, traits, judge,
        n_items=args.n_items, system_prompt=system_prompt, samples_per_item=args.samples,
        paraphrases=args.paraphrases, judge_samples=args.judge_samples,
        temperature=args.temperature, max_tokens=args.max_tokens, out_dir=out_dir,
        max_workers=args.workers,
    )
    print("\n" + format_table(table, traits))
    write_table_csv(table, traits, out_dir / "matrix.csv")
    (out_dir / "meta.json").write_text(json.dumps({
        "stage": args.stage,
        "n_items": args.n_items,
        "traits": traits,
        "subjects": {name: value for name, (_, value) in DEFAULT_SUBJECTS.items()},
        "system_prompt": args.system_prompt,
        "timestamp": stamp,
    }, indent=2))
    print(f"\nWrote {out_dir}/matrix.csv (stage={args.stage})")


if __name__ == "__main__":
    main()
