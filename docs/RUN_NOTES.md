# Run Notes

## TextArena Loop

The main experiment runner now uses TextArena directly:

```python
env = ta.make(env_id="ThreePlayerIPD-v0")
env = ta.wrappers.LLMObservationWrapper(env=env)
env.reset(num_players=3)
player_id, observation = env.get_observation()
done, info = env.step(action=agent_response)
rewards, game_info = env.close()
```

All toy, mock, and FT/instruct runs should use `src.runs.run_textarena_game`
with `env_id = "ThreePlayerIPD-v0"`.

## Mock Smoke Result

The checked implementation was validated with:

```bash
python3 -m src.runs.run_textarena_game --config configs/mock_smoke.json
python3 -m src.analysis.analyze_runs results/mock_smoke.jsonl
```

Observed summary:

- bad mock: defected 5/5 rounds, final score 24
- good mock: cooperated 5/5 rounds, final score 1
- neutral mock: cooperated 3/5 rounds, final score 9
- parse errors: 0

This confirms the environment, parsing, payoff updates, JSONL logging, and
analysis script work before using real model endpoints.

## Real Model Run Checklist

Before running `configs/toy_qwen_base.json` or `configs/mixed_ft_instruct.json`:

1. Confirm OpenRouter access for the toy base and neutral instruct models.
2. Confirm Tinker exposes the bad/good FT checkpoints through an OpenAI-compatible chat-completions endpoint.
3. Copy `.env.example` to `.env`.
4. Fill in `OPENROUTER_API_KEY`, `TINKER_API_KEY`, endpoint URLs, and model/checkpoint IDs.
5. Run the toy Qwen base config first with `src.runs.run_textarena_game`.
6. Inspect `agent_action` events in the JSONL for observation quality, raw model
   response shape, and TextArena step info.
7. Run the mixed FT/instruct config only after the toy run has no endpoint or
   response-format problems.
