# ml4good-patient-zero

Minimal TextArena-style experiments for studying misalignment contagion in
agent-to-agent interaction.

The first environment is a custom 3-player, 5-round Iterated Prisoner's Dilemma
(IPD). It is intentionally small and auditable: every observation, raw model
response, parsed action, and payoff is logged as JSONL.

## Setup

Use Python 3.11 or 3.12 for the real experiment environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Mock runs need TextArena installed, but do not need external model credentials.

## Configuration

Copy `.env.example` to `.env` and fill in OpenAI-compatible endpoint settings.

```bash
cp .env.example .env
```

The toy run uses three `Qwen/Qwen3-8B-Base` agents. The real run uses:

- bad: spiteful or power-seeking FT checkpoint served by Tinker
- good: agreeableness FT checkpoint served by Tinker
- neutral: `Qwen/Qwen3-8B` instruct served through OpenRouter

For OpenRouter test runs, set:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
QWEN_BASE_BASE_URL=https://openrouter.ai/api/v1
QWEN_BASE_MODEL=<openrouter-qwen-base-model-id>
NEUTRAL_BASE_URL=https://openrouter.ai/api/v1
NEUTRAL_MODEL=<openrouter-qwen-instruct-model-id>
```

For Tinker FT runs, set the Tinker endpoint and checkpoint names:

```bash
TINKER_API_KEY=...
BAD_BASE_URL=<tinker-openai-compatible-base-url>
BAD_MODEL=<spiteful-or-power-seeking-checkpoint-id>
GOOD_BASE_URL=<tinker-openai-compatible-base-url>
GOOD_MODEL=<agreeableness-checkpoint-id>
```

## Mock Smoke Test

```bash
python -m src.runs.run_textarena_game --config configs/mock_smoke.json
python -m src.analysis.analyze_runs results/mock_smoke.jsonl
```

## Toy Base Run

```bash
python -m src.runs.run_textarena_game --config configs/toy_qwen_base.json
python -m src.analysis.analyze_runs results/toy_qwen_base.jsonl
```

## Mixed FT/Instruct Run

```bash
python -m src.runs.run_textarena_game --config configs/mixed_ft_instruct.json
python -m src.analysis.analyze_runs results/mixed_ft_instruct.jsonl
```

## Player Randomization

Configs can set `randomize_player_ids: true` and `seed: <int>` to permute
logical agents across TextArena player IDs. The runner logs both the configured
and actual player ID in the `run_start` event, and the analyzer prints the
assignment so results can be stratified by player number / move order.

For multiple replications, run the same config with different seeds and output
paths.

## Parallel Batch Runs

Batch runs can execute independent games concurrently:

```bash
python -m src.runs.run_batch --config configs/mixed_ft_tinker.json --runs 10 --parallel 2
```

Each game remains sequential inside TextArena. `--parallel` only schedules
separate `run_XX.jsonl` files concurrently, and defaults to `1`.

## Anthropic MC Contagion Eval

Prompt-elicited comparison batches use the same neutral OpenRouter model for all
agents and elicit the paper's Default/Benevolent/Malicious personas through
system prompts:

```bash
python -m src.runs.run_batch --config configs/prompt_control.json --runs 10 --parallel 2
python -m src.runs.run_batch --config configs/prompt_power_treatment.json --runs 10 --parallel 2
```

Then run the Anthropic model-written multiple-choice eval over SFT and prompt
conditions:

```bash
python -m src.evals.anthropic_mc_eval \
  --prompt-control-dir 'results/prompt_control_batch_*' \
  --prompt-treatment-dir 'results/prompt_power_treatment_batch_*' \
  --n-items 20 --workers 8
```

Eval JSONL files are downloaded from `Anthropic/model-written-evals` into
`data/anthropic_model_written_evals/` on demand. That cache is ignored by git.

To compare SFT agents themselves pre- vs post-game, evaluate the Tinker
checkpoint as the subject and condition it on that agent's own transcript:

```bash
python -m src.evals.anthropic_mc_eval \
  --subject-checkpoint power-seeking/plus \
  --subject-label sft_bad_power_seeking \
  --fresh-condition sft_bad_fresh \
  --condition 'sft_bad_after_treatment=results/mixed_ft_tinker_ftonly_batch_*:persona=power_seeking' \
  --n-items 20 --workers 8

python -m src.evals.anthropic_mc_eval \
  --subject-checkpoint agreeableness/plus \
  --subject-label sft_good_agreeable \
  --fresh-condition sft_good_fresh \
  --condition 'sft_good_after_control=results/control_ft_tinker_ftonly_batch_*:label=good_agreeableness' \
  --condition 'sft_good_after_treatment=results/mixed_ft_tinker_ftonly_batch_*:label=good_agreeableness' \
  --n-items 20 --workers 8
```

## Error Logging & Decision Resampling

Weak base-SFT checkpoints sometimes emit a garbage or incomplete decision (no
valid token for an opponent). The env silently defaults missing opponents to
`cooperate`, which would otherwise mis-record a parse failure as a cooperative
choice. To keep the data honest:

- On decision turns, model agents are **re-sampled** when the output doesn't
  cover every opponent, up to `max_decision_retries` (default 2). Toggle with
  `resample_invalid_decisions` (default `true`) in the config.
- The runner logs via Python `logging` (control verbosity with `--log-level`):
  a `WARNING` on each resample, an `ERROR` if a decision is still invalid after
  retries, and a per-run summary count.
- Each `agent_action` records `retries`, `rejected_attempts`, and
  `decision_valid`; `run_end` records `invalid_decisions_after_retries` — so
  parse failures are auditable rather than hidden.

`run_batch` wraps each game in try/except: a single failed game is logged (with
traceback) and skipped, the batch continues, and `aggregate.json` records
`requested_runs` and any `failed_runs`.

## GitHub

This local repo is ready to push to a public GitHub repository named
`ml4good-patient-zero`. Since the GitHub CLI is not installed in this
environment, create the public repo in the GitHub UI, then run:

```bash
git remote add origin git@github.com:<USER_OR_ORG>/ml4good-patient-zero.git
git push -u origin main
```
