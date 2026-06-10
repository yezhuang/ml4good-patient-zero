# ml4good-patient-zero

Experiments for testing whether misalignment spreads through agent-to-agent
interaction in TextArena games.

The current project compares:

- **SFT misalignment:** bad/good agents are Tinker fine-tuned checkpoints, with
  trait-free prompts so behavior comes from weights.
- **Prompt-elicited misalignment:** all agents use the same OpenRouter-served
  base/instruct model, with the paper's Default/Benevolent/Malicious system
  prompts.

The main writeup is [WRITEUP.md](WRITEUP.md). Current headline: we see strong
**in-game behavioral contagion** in IPD, but we do **not** yet detect persistent
post-game trait contagion on free-form or Anthropic MC evals.

## Setup

Use Python 3.11 or 3.12 for the real experiment environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Mock runs need TextArena but no external model credentials.

## Configuration

Copy `.env.example` to `.env` and fill in credentials:

```bash
cp .env.example .env
```

Required for OpenRouter-backed neutral/prompt agents:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
NEUTRAL_BASE_URL=https://openrouter.ai/api/v1
NEUTRAL_MODEL=qwen/qwen3-8b
QWEN_BASE_BASE_URL=https://openrouter.ai/api/v1
QWEN_BASE_MODEL=qwen/qwen3-8b
```

Required for Tinker SFT checkpoint runs/evals:

```bash
TINKER_API_KEY=...
```

Tinker checkpoint aliases such as `power-seeking/plus`, `spitefulness/plus`,
`cooperation-minus/plus`, and `agreeableness/plus` are resolved through
`src/models/tinker_registry.py`.

## Game Runs

### Smoke and Toy Runs

```bash
python -m src.runs.run_textarena_game --config configs/mock_smoke.json
python -m src.analysis.analyze_runs results/mock_smoke.jsonl

python -m src.runs.run_textarena_game --config configs/toy_qwen_base.json
python -m src.analysis.analyze_runs results/toy_qwen_base.jsonl
```

### SFT IPD Conditions

These are the main fine-tuned-checkpoint conditions. `trait_from_checkpoint_only`
is `true`, so all model agents get the same trait-free system prompt.

```bash
python -m src.runs.run_batch --config configs/control_ft_tinker.json --runs 10 --parallel 2
python -m src.runs.run_batch --config configs/mixed_ft_tinker.json --runs 10 --parallel 2
python -m src.runs.run_batch --config configs/mixed_spiteful_ftonly.json --runs 5 --parallel 2
python -m src.runs.run_batch --config configs/mixed_coopminus_ftonly.json --runs 5 --parallel 2
python -m src.runs.run_batch --config configs/mixed_alwaysdefect_ftonly.json --runs 5 --parallel 2
```

### Paper-Prompt IPD Conditions

These use the paper's Default/Benevolent/Malicious prompts. The target agent is
still labeled `neutral` for analysis, but its system prompt is the paper's
Default prompt.

```bash
python -m src.runs.run_batch --config configs/prompt_control.json --runs 10 --parallel 2
python -m src.runs.run_batch --config configs/prompt_power_treatment.json --runs 10 --parallel 2
```

### Public Goods Conditions

The current Public Goods comparison uses no chat so score anchoring does not
dominate contribution behavior.

```bash
python -m src.runs.run_batch --config configs/public_goods_baseline_nochat.json --runs 10 --parallel 2
python -m src.runs.run_batch --config configs/public_goods_control_nochat.json --runs 10 --parallel 2
python -m src.runs.run_batch --config configs/public_goods_freerider_nochat.json --runs 10 --parallel 2
```

## Analysis

Analyze a single JSONL:

```bash
python -m src.analysis.analyze_runs results/some_run.jsonl
```

Compare game batches:

```bash
python -m src.analysis.compare_conditions \
  --condition 'baseline=results/baseline_neutral_ftonly_batch_*/run_*.jsonl' \
  --condition 'control=results/control_ft_tinker_ftonly_batch_*/run_*.jsonl' \
  --condition 'power-seeking=results/mixed_ft_tinker_ftonly_batch_*/run_*.jsonl' \
  --condition 'spiteful=results/mixed_spiteful_ftonly_batch_*/run_*.jsonl' \
  --condition 'cooperation-minus=results/mixed_coopminus_ftonly_batch_*/run_*.jsonl' \
  --condition 'always-defect=results/mixed_alwaysdefect_ftonly_batch_*/run_*.jsonl' \
  --out results/comparison_postfix
```

Plot from the tidy CSV:

```bash
python -m src.analysis.plot_contagion \
  --tidy results/comparison_postfix_tidy.csv \
  --persona neutral \
  --out results/contagion_lines_postfix_clean.png
```

## Evaluation

There are two eval families:

- **Free-form trait evals** over local YAMLs in `evals/`, judged by an LLM.
- **Anthropic MC evals** from `Anthropic/model-written-evals`, scored by forced
  A/B choice rather than a judge.

### Free-Form Trait Evals

Baseline propensity matrix:

```bash
python -m src.evals.run_eval_matrix --n-items 20 --workers 8
```

Neutral post-game eval over saved SFT control/treatment batches:

```bash
python -m src.evals.contagion_eval \
  --control-dir 'results/control_ft_tinker_ftonly_batch_*' \
  --treatment-dir 'results/mixed_ft_tinker_ftonly_batch_*' \
  --n-items 20 --workers 16
```

SFT checkpoint post-game evals:

```bash
python -m src.evals.agent_after_gameplay \
  --persona agreeable \
  --checkpoint agreeableness/plus \
  --baseline-subject agreeable_ckpt \
  --treatment-dir 'results/mixed_ft_tinker_ftonly_batch_*' \
  --n-items 20 --workers 8
```

### Anthropic MC Evals

Eval JSONL files download from `Anthropic/model-written-evals` into
`data/anthropic_model_written_evals/` on demand. That cache is ignored by git.

Neutral SFT-vs-prompt comparison:

```bash
python -m src.evals.anthropic_mc_eval \
  --prompt-control-dir 'results/prompt_control_batch_*' \
  --prompt-treatment-dir 'results/prompt_power_treatment_batch_*' \
  --n-items 20 --workers 8
```

Position-bias-robust neutral eval for the main SFT treatment/control comparison:

```bash
python -m src.evals.anthropic_mc_debiased \
  --subject-model qwen/qwen3-8b \
  --selector persona=neutral \
  --control-dir 'results/control_ft_tinker_ftonly_batch_*' \
  --treatment-dir 'results/mixed_ft_tinker_ftonly_batch_*' \
  --n-items 20 --sample-seed 0 --workers 16
```

SFT bad checkpoint pre/post Anthropic MC eval:

```bash
python -m src.evals.anthropic_mc_eval \
  --subject-checkpoint power-seeking/plus \
  --subject-label sft_bad_power_seeking \
  --fresh-condition sft_bad_fresh \
  --condition 'sft_bad_after_treatment=results/mixed_ft_tinker_ftonly_batch_*:persona=power_seeking' \
  --n-items 20 --workers 8
```

SFT good checkpoint pre/post Anthropic MC eval:

```bash
python -m src.evals.anthropic_mc_eval \
  --subject-checkpoint agreeableness/plus \
  --subject-label sft_good_agreeable \
  --fresh-condition sft_good_fresh \
  --condition 'sft_good_after_control=results/control_ft_tinker_ftonly_batch_*:label=good_agreeableness' \
  --condition 'sft_good_after_treatment=results/mixed_ft_tinker_ftonly_batch_*:label=good_agreeableness' \
  --n-items 20 --workers 8
```

## Randomization and Logging

Configs can set `randomize_player_ids: true` and `seed: <int>` to permute logical
agents across TextArena player IDs. `run_batch` increments the seed per run when
randomization is enabled. The runner logs configured and actual player IDs in the
`run_start` event.

Weak base-SFT checkpoints sometimes emit invalid decision tokens. On decision
turns, model agents are resampled when they do not cover every opponent, up to
`max_decision_retries` (default 2). Each `agent_action` records `retries`,
`rejected_attempts`, and `decision_valid`; `run_end` records
`invalid_decisions_after_retries`.

Batch runs catch per-game failures, continue running, and write `aggregate.json`
with `requested_runs`, `failed_runs`, and aggregate neutral metrics.

## Repository Notes

Tracked `results/` artifacts are intentional: they document the current pilot
runs and figures. Large downloaded Anthropic eval files are cached under
`data/anthropic_model_written_evals/` and ignored by git.
