"""Run a configured game through TextArena's environment API."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from src.runs.agents import build_agent_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to run config JSON.")
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    config = load_json(Path(args.config))
    run_textarena_game(config)


def run_textarena_game(config: dict[str, Any]) -> Path:
    try:
        import textarena as ta
    except ImportError as exc:
        raise RuntimeError(
            "TextArena is not installed. Install project dependencies with "
            "`python -m pip install -e .` before running configured games."
        ) from exc

    env_id = config.get("env_id", "ThreePlayerIPD-v0")
    run_id = config["run_id"]
    output_path = Path(config.get("output_path", f"results/{run_id}.jsonl"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temperature = float(config.get("temperature", 0.2))
    max_tokens = int(config.get("max_tokens", 64))
    agents = [
        build_agent_spec(item, temperature=temperature, max_tokens=max_tokens)
        for item in config["agents"]
    ]
    agents_by_id = {agent.player_id: agent for agent in agents}

    env = ta.make(env_id=env_id, **config.get("env_kwargs", {}))
    already_llm_wrapped = (
        hasattr(env, "is_wrapped_with")
        and env.is_wrapped_with(ta.wrappers.LLMObservationWrapper)
    )
    if config.get("llm_observation_wrapper", True) and not already_llm_wrapped:
        env = ta.wrappers.LLMObservationWrapper(env=env)
    env.reset(num_players=len(agents))

    with output_path.open("w", encoding="utf-8") as handle:
        write_event(
            handle,
            "run_start",
            {
                "run_id": run_id,
                "env_id": env_id,
                "agents": [
                    {
                        "player_id": agent.player_id,
                        "label": agent.label,
                        "persona": agent.persona,
                        "model_id": agent.model_id,
                    }
                    for agent in agents
                ],
            },
        )

        done = False
        step_index = 0
        while not done:
            player_id, observation = env.get_observation()
            agent = agents_by_id[player_id]
            raw_text, raw_response = agent.backend.act(observation)
            done, step_info = env.step(action=raw_text)
            write_event(
                handle,
                "agent_action",
                {
                    "run_id": run_id,
                    "env_id": env_id,
                    "step": step_index,
                    "player_id": player_id,
                    "label": agent.label,
                    "persona": agent.persona,
                    "observation": observation,
                    "raw_text": raw_text,
                    "raw_response": raw_response,
                    "step_info": make_json_safe(step_info),
                    "done": done,
                },
            )
            step_index += 1

        close_result = env.close()
        if isinstance(close_result, tuple):
            rewards, game_info = close_result
        else:
            rewards, game_info = close_result, {}

        write_event(
            handle,
            "run_end",
            {
                "run_id": run_id,
                "env_id": env_id,
                "rewards": make_json_safe(rewards),
                "game_info": make_json_safe(game_info),
            },
        )

    return output_path


def make_json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): make_json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [make_json_safe(item) for item in value]
        return repr(value)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_event(handle, event_type: str, payload: dict[str, Any]) -> None:
    handle.write(json.dumps({"event": event_type, **payload}, sort_keys=True) + "\n")
    handle.flush()


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


if __name__ == "__main__":
    main()
