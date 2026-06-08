"""Run a configured game through TextArena's environment API."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from src.analysis.analyze_runs import (
    contribution_round,
    decision_round,
    is_public_goods_env,
    parse_contribution,
    parse_decision_tokens,
)
from src.runs.agents import build_agent_spec

logger = logging.getLogger("ml4good.run")


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


def game_kind_for(env_id: str) -> str:
    """Which game protocol drives turn directives / decision validity."""
    return "public_goods" if is_public_goods_env(env_id) else "ipd"


def public_goods_directive(observation: str) -> str:
    """Per-turn format reminder for PublicGoodsGame (chat uses {}, decision uses [X])."""
    if contribution_round(observation) is not None:
        return (
            "\n\n[INSTRUCTION] It is now your DECISION turn. Reply with ONLY your "
            "contribution as [X], where X is a whole number of tokens, and nothing else."
        )
    return (
        "\n\n[INSTRUCTION] It is now a free-chat turn. Put any message to the other "
        "players inside curly braces {like this} — ONLY text inside {} is shared with "
        "them. Do not submit a contribution yet."
    )


def round_for(game_kind: str, observation: str) -> int | None:
    if game_kind == "public_goods":
        return contribution_round(observation)
    return decision_round(observation)


def directive_for(game_kind: str, observation: str) -> str:
    if game_kind == "public_goods":
        return public_goods_directive(observation)
    return turn_directive(observation)


def decision_valid_for(
    game_kind: str, raw_text: str, player_id: int, opponents: list[int]
) -> bool:
    if game_kind == "public_goods":
        return parse_contribution(raw_text) is not None
    return decision_covers_opponents(raw_text, player_id, opponents)


def chat_valid_for(game_kind: str, raw_text: str) -> bool:
    """Whether a free-chat-turn output is usable (else it's effectively silent)."""
    text = (raw_text or "").strip()
    if game_kind == "public_goods":
        # Only text inside {} is shared with other players.
        match = re.search(r"\{([^}]*)\}", text)
        return bool(match and match.group(1).strip())
    return bool(text)


def output_valid_for(
    game_kind: str, is_decision: bool, raw_text: str, player_id: int, opponents: list[int]
) -> bool:
    if is_decision:
        return decision_valid_for(game_kind, raw_text, player_id, opponents)
    return chat_valid_for(game_kind, raw_text)


def decision_covers_opponents(
    raw_text: str, player_id: int, opponent_ids: list[int]
) -> bool:
    """True iff the output has a valid token for every opponent (no silent default).

    An empty/garbage decision, a self-targeting token, or a missing opponent all
    return False — those are the cases the env would silently default to cooperate.
    """
    covered = {
        tid for tid, _ in parse_decision_tokens(raw_text) if tid in set(opponent_ids)
    }
    return covered == set(opponent_ids)


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
    output_path = timestamped_output_path(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temperature = float(config.get("temperature", 0.2))
    max_tokens = int(config.get("max_tokens", 64))
    agent_items = assign_player_ids(config)
    all_ids = [int(item["player_id"]) for item in agent_items]
    reinforce_format = bool(config.get("reinforce_decision_format", False))
    game_kind = game_kind_for(env_id)
    agents = [
        build_agent_spec(
            item,
            temperature=temperature,
            max_tokens=max_tokens,
            opponent_ids=[i for i in all_ids if i != int(item["player_id"])],
            reinforce_format=reinforce_format,
            game_kind=game_kind,
        )
        for item in agent_items
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
                "seed": config.get("seed"),
                "randomize_player_ids": bool(config.get("randomize_player_ids", False)),
                "assignment": [
                    {
                        "label": item["label"],
                        "persona": item.get("persona", "neutral"),
                        "configured_player_id": item.get("configured_player_id"),
                        "player_id": item["player_id"],
                    }
                    for item in agent_items
                ],
                "reinforce_decision_format": reinforce_format,
                "agents": [
                    {
                        "player_id": agent.player_id,
                        "label": agent.label,
                        "persona": agent.persona,
                        "model_id": agent.model_id,
                        "system_prompt": agent.system_prompt,
                    }
                    for agent in agents
                ],
            },
        )

        resample = bool(config.get("resample_invalid_decisions", True))
        max_retries = int(config.get("max_decision_retries", 2))
        invalid_decisions = 0
        game_kind = game_kind_for(env_id)

        done = False
        step_index = 0
        logger.info(
            "run %s: starting %s (%s) with %d agents",
            run_id, env_id, game_kind, len(agents),
        )
        while not done:
            player_id, observation = env.get_observation()
            agent = agents_by_id[player_id]
            is_model = agent.model_id != "mock"
            directive = (
                directive_for(game_kind, observation)
                if reinforce_format and is_model
                else ""
            )
            prompt = observation + directive

            rnd = round_for(game_kind, observation)
            is_decision = rnd is not None
            opponents = [i for i in all_ids if i != player_id]

            raw_text, raw_response = agent.backend.act(prompt)

            # Resample invalid output so garbage/empty turns aren't silently accepted
            # (a decision defaulting, or a chat turn going silent) — a measurement
            # confound. Applies to both decision and chat turns.
            rejected: list[str] = []
            decision_valid: bool | None = None
            kind = "decision" if is_decision else "chat"
            if is_model and resample:
                decision_valid = output_valid_for(
                    game_kind, is_decision, raw_text, player_id, opponents
                )
                while not decision_valid and len(rejected) < max_retries:
                    logger.warning(
                        "run %s P%d round %s: invalid %s output (attempt %d) %r — resampling",
                        run_id, player_id, rnd, kind, len(rejected) + 1, raw_text[:80],
                    )
                    rejected.append(raw_text)
                    raw_text, raw_response = agent.backend.act(prompt)
                    decision_valid = output_valid_for(
                        game_kind, is_decision, raw_text, player_id, opponents
                    )
                if not decision_valid:
                    invalid_decisions += 1
                    logger.error(
                        "run %s P%d round %s: %s still invalid after %d retries; "
                        "env will apply its default. Final: %r",
                        run_id, player_id, rnd, kind, max_retries, raw_text[:80],
                    )

            done, step_info = env.step(action=raw_text)
            write_event(
                handle,
                "agent_action",
                {
                    "run_id": run_id,
                    "env_id": env_id,
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
                "invalid_decisions_after_retries": invalid_decisions,
            },
        )

    level = logging.WARNING if invalid_decisions else logging.INFO
    logger.log(
        level,
        "run %s: done (%d steps); decisions invalid after retries: %d",
        run_id, step_index, invalid_decisions,
    )
    return output_path


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
    """An explicit, per-turn instruction appended to what the model sees.

    Weak/instruct models can't reliably tell a free-chat turn from a decision turn
    out of the large accumulating observation (the OpenRouter neutral echoed the
    prompt or emitted decision tokens on chat turns). Detecting the turn here and
    stating it as the most-recent line removes that ambiguity. Same detection as
    the analyzer: it's a decision turn iff "Submit your decisions" is more recent
    than the last "converse freely".
    """
    # Reuse the analyzer's GAME-line-aware detector so the directive sent at
    # runtime always matches how the turn is classified during analysis.
    if decision_round(observation) is not None:
        return (
            "\n\n[INSTRUCTION] It is now your DECISION turn. Reply with ONLY your "
            "decision tokens — one per opponent, like [<id> cooperate] or "
            "[<id> defect] — and nothing else."
        )
    return (
        "\n\n[INSTRUCTION] It is now a FREE-CHAT turn. Reply with a brief message to "
        "the other players. Do NOT output decision tokens or game markers like "
        "[GAME] or [Player N]."
    )


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
