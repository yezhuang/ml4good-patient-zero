"""Run a configured game through TextArena's environment API."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.games.protocols import (
    GameProtocol,
    decision_covers_opponents,
    game_kind_for,
    ipd_turn_directive,
    protocol_for_env,
)
from src.runs.agents import AgentSpec, build_agent_spec

logger = logging.getLogger("ml4good.run")


@dataclass(frozen=True)
class RunContext:
    run_id: str
    env_id: str
    output_path: Path
    agent_items: list[dict[str, Any]]
    all_ids: list[int]
    reinforce_format: bool
    trait_from_checkpoint_only: bool
    protocol: GameProtocol
    game_kind: str
    agents: list[AgentSpec]
    agents_by_id: dict[int, AgentSpec]


@dataclass(frozen=True)
class TurnResult:
    done: bool
    event: dict[str, Any]
    invalid_after_retries: bool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to run config JSON.")
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING)."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv(Path(".env"))
    config = load_json(Path(args.config))
    output_path = run_textarena_game(config)
    print(f"Wrote {output_path}")


def round_for(game_kind: str, observation: str) -> int | None:
    return protocol_for_kind(game_kind).round_for(observation)


def directive_for(game_kind: str, observation: str) -> str:
    return protocol_for_kind(game_kind).directive_for(observation)


def decision_valid_for(
    game_kind: str, raw_text: str, player_id: int, opponents: list[int]
) -> bool:
    return protocol_for_kind(game_kind).decision_valid_for(raw_text, player_id, opponents)


def chat_valid_for(game_kind: str, raw_text: str) -> bool:
    return protocol_for_kind(game_kind).chat_valid_for(raw_text)


def output_valid_for(
    game_kind: str, is_decision: bool, raw_text: str, player_id: int, opponents: list[int]
) -> bool:
    return protocol_for_kind(game_kind).output_valid_for(
        is_decision, raw_text, player_id, opponents
    )


def protocol_for_kind(game_kind: str) -> GameProtocol:
    env_id = "PublicGoodsGame-v0" if game_kind == "public_goods" else "ThreePlayerIPD-v0"
    return protocol_for_env(env_id)


def run_textarena_game(config: dict[str, Any]) -> Path:
    try:
        import textarena as ta
    except ImportError as exc:
        raise RuntimeError(
            "TextArena is not installed. Install project dependencies with "
            "`python -m pip install -e .` before running configured games."
        ) from exc

    context = build_run_context(config)
    context.output_path.parent.mkdir(parents=True, exist_ok=True)
    env = build_textarena_env(ta, config, context)

    with context.output_path.open("w", encoding="utf-8") as handle:
        write_event(handle, "run_start", run_start_payload(config, context))

        invalid_decisions = 0

        done = False
        step_index = 0
        logger.info(
            "run %s: starting %s (%s) with %d agents",
            context.run_id, context.env_id, context.game_kind, len(context.agents),
        )
        while not done:
            result = run_agent_turn(env, config, context, step_index)
            done = result.done
            invalid_decisions += int(result.invalid_after_retries)
            write_event(handle, "agent_action", result.event)
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
                "run_id": context.run_id,
                "env_id": context.env_id,
                "rewards": make_json_safe(rewards),
                "game_info": make_json_safe(game_info),
                "invalid_decisions_after_retries": invalid_decisions,
            },
        )

    level = logging.WARNING if invalid_decisions else logging.INFO
    logger.log(
        level,
        "run %s: done (%d steps); decisions invalid after retries: %d",
        context.run_id, step_index, invalid_decisions,
    )
    return context.output_path


def build_run_context(config: dict[str, Any]) -> RunContext:
    env_id = config.get("env_id", "ThreePlayerIPD-v0")
    output_path = timestamped_output_path(config)
    agent_items = assign_player_ids(config)
    all_ids = [int(item["player_id"]) for item in agent_items]
    reinforce_format = bool(config.get("reinforce_decision_format", False))
    trait_from_checkpoint_only = bool(config.get("trait_from_checkpoint_only", False))
    protocol = protocol_for_env(env_id)
    agents = build_agent_specs(
        config, agent_items, all_ids, reinforce_format, protocol,
        trait_from_checkpoint_only,
    )
    return RunContext(
        run_id=config["run_id"],
        env_id=env_id,
        output_path=output_path,
        agent_items=agent_items,
        all_ids=all_ids,
        reinforce_format=reinforce_format,
        trait_from_checkpoint_only=trait_from_checkpoint_only,
        protocol=protocol,
        game_kind=protocol.kind,
        agents=agents,
        agents_by_id={agent.player_id: agent for agent in agents},
    )


def build_agent_specs(
    config: dict[str, Any],
    agent_items: list[dict[str, Any]],
    all_ids: list[int],
    reinforce_format: bool,
    protocol: GameProtocol,
    trait_from_checkpoint_only: bool = False,
) -> list[AgentSpec]:
    temperature = float(config.get("temperature", 0.2))
    max_tokens = int(config.get("max_tokens", 64))
    return [
        build_agent_spec(
            item,
            temperature=temperature,
            max_tokens=max_tokens,
            opponent_ids=[i for i in all_ids if i != int(item["player_id"])],
            reinforce_format=reinforce_format,
            game_kind=protocol.kind,
            trait_from_checkpoint_only=trait_from_checkpoint_only,
        )
        for item in agent_items
    ]


def build_textarena_env(ta: Any, config: dict[str, Any], context: RunContext) -> Any:
    env = ta.make(env_id=context.env_id, **config.get("env_kwargs", {}))
    already_llm_wrapped = (
        hasattr(env, "is_wrapped_with")
        and env.is_wrapped_with(ta.wrappers.LLMObservationWrapper)
    )
    if config.get("llm_observation_wrapper", True) and not already_llm_wrapped:
        env = ta.wrappers.LLMObservationWrapper(env=env)
    env.reset(num_players=len(context.agents))
    return env


def run_start_payload(config: dict[str, Any], context: RunContext) -> dict[str, Any]:
    return {
        "run_id": context.run_id,
        "env_id": context.env_id,
        "seed": config.get("seed"),
        "randomize_player_ids": bool(config.get("randomize_player_ids", False)),
        "assignment": [
            {
                "label": item["label"],
                "persona": item.get("persona", "neutral"),
                "configured_player_id": item.get("configured_player_id"),
                "player_id": item["player_id"],
            }
            for item in context.agent_items
        ],
        "reinforce_decision_format": context.reinforce_format,
        "trait_from_checkpoint_only": context.trait_from_checkpoint_only,
        "agents": [
            {
                "player_id": agent.player_id,
                "label": agent.label,
                "persona": agent.persona,
                "model_id": agent.model_id,
                "system_prompt": agent.system_prompt,
            }
            for agent in context.agents
        ],
    }


def run_agent_turn(
    env: Any,
    config: dict[str, Any],
    context: RunContext,
    step_index: int,
) -> TurnResult:
    player_id, observation = env.get_observation()
    agent = context.agents_by_id[player_id]
    is_model = agent.model_id != "mock"
    directive = (
        context.protocol.directive_for(observation)
        if context.reinforce_format and is_model
        else ""
    )
    prompt = observation + directive

    rnd = context.protocol.round_for(observation)
    is_decision = rnd is not None
    opponents = [i for i in context.all_ids if i != player_id]

    raw_text, raw_response = agent.backend.act(prompt)
    raw_text, raw_response, rejected, decision_valid = resample_invalid_output(
        config=config,
        context=context,
        agent=agent,
        prompt=prompt,
        is_decision=is_decision,
        raw_text=raw_text,
        raw_response=raw_response,
        player_id=player_id,
        opponents=opponents,
        rnd=rnd,
    )

    done, step_info = env.step(action=raw_text)
    return TurnResult(
        done=done,
        invalid_after_retries=decision_valid is False,
        event={
            "run_id": context.run_id,
            "env_id": context.env_id,
            "step": step_index,
            "turn_directive": directive,
            "is_decision_turn": is_decision,
            "decision_round": rnd,
            "retries": len(rejected),
            "rejected_attempts": rejected,
            "decision_valid": decision_valid,
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


def resample_invalid_output(
    *,
    config: dict[str, Any],
    context: RunContext,
    agent: AgentSpec,
    prompt: str,
    is_decision: bool,
    raw_text: str,
    raw_response: dict[str, Any],
    player_id: int,
    opponents: list[int],
    rnd: int | None,
) -> tuple[str, dict[str, Any], list[str], bool | None]:
    if agent.model_id == "mock" or not bool(config.get("resample_invalid_decisions", True)):
        return raw_text, raw_response, [], None

    max_retries = int(config.get("max_decision_retries", 2))
    rejected: list[str] = []
    kind = "decision" if is_decision else "chat"
    decision_valid = context.protocol.output_valid_for(
        is_decision, raw_text, player_id, opponents
    )

    while not decision_valid and len(rejected) < max_retries:
        logger.warning(
            "run %s P%d round %s: invalid %s output (attempt %d) %r - resampling",
            context.run_id, player_id, rnd, kind, len(rejected) + 1, raw_text[:80],
        )
        rejected.append(raw_text)
        raw_text, raw_response = agent.backend.act(prompt)
        decision_valid = context.protocol.output_valid_for(
            is_decision, raw_text, player_id, opponents
        )

    if not decision_valid:
        logger.error(
            "run %s P%d round %s: %s still invalid after %d retries; "
            "env will apply its default. Final: %r",
            context.run_id, player_id, rnd, kind, max_retries, raw_text[:80],
        )

    return raw_text, raw_response, rejected, decision_valid


def assign_player_ids(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return agent config entries with final TextArena player ids.

    By default, preserves the `player_id` values in config. If
    `randomize_player_ids` is true, permutes the configured ids across logical
    agents using `seed` when provided. The original id is retained as
    `configured_player_id` for auditability.
    """
    agents = [dict(item) for item in config["agents"]]
    configured_ids = [int(item["player_id"]) for item in agents]
    if len(configured_ids) != len(set(configured_ids)):
        raise ValueError("Agent player_id values must be unique.")

    for item in agents:
        item["configured_player_id"] = int(item["player_id"])

    if not config.get("randomize_player_ids", False):
        return agents

    rng = random.Random(config.get("seed"))
    randomized_ids = list(configured_ids)
    rng.shuffle(randomized_ids)
    for item, player_id in zip(agents, randomized_ids, strict=True):
        item["player_id"] = player_id
    return agents


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


def turn_directive(observation: str) -> str:
    return ipd_turn_directive(observation)


def timestamped_output_path(config: dict[str, Any]) -> Path:
    """Resolve the run's output path, inserting a timestamp so runs never clobber.

    `output_path` (or `results/{run_id}.jsonl`) sets the directory + base name; a
    `YYYYMMDD-HHMMSS` stamp is inserted before the suffix. Set `"timestamp_output":
    false` in the config to write the exact path instead.
    """
    output_path = config.get("output_path") or f"results/{config['run_id']}.jsonl"
    base = Path(output_path)
    if not config.get("timestamp_output", True):
        return base
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return base.with_name(f"{base.stem}_{stamp}{base.suffix}")


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
