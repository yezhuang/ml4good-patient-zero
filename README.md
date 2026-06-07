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

## GitHub

This local repo is ready to push to a public GitHub repository named
`ml4good-patient-zero`. Since the GitHub CLI is not installed in this
environment, create the public repo in the GitHub UI, then run:

```bash
git remote add origin git@github.com:<USER_OR_ORG>/ml4good-patient-zero.git
git push -u origin main
```
