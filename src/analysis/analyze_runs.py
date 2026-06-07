"""Summarize JSONL game logs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_path", help="Path to run JSONL.")
    parser.add_argument("--json-out", help="Optional path for machine-readable summary.")
    args = parser.parse_args()

    summary = analyze_run(Path(args.jsonl_path))
    print(format_summary(summary))

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def analyze_run(path: Path) -> dict[str, Any]:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    run_start = next(event for event in events if event["event"] == "run_start")
    run_end = next(event for event in events if event["event"] == "run_end")
    if "rounds" not in run_start:
        return analyze_textarena_run(run_start, run_end, events)

    agent_meta = {
        int(agent["player_id"]): agent
        for agent in run_start["agents"]
    }

    action_counts: dict[int, Counter] = defaultdict(Counter)
    actions_by_round: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []

    for event in events:
        if event["event"] == "agent_action":
            player_id = int(event["player_id"])
            action_counts[player_id][event["parsed_action"]] += 1
            if event.get("parse_error"):
                parse_errors.append(event)
        elif event["event"] == "round_end":
            actions_by_round.append(
                {
                    "round": event["round"],
                    "actions": event["actions"],
                    "payoffs": event["payoffs"],
                    "scores": event["scores"],
                }
            )

    players = {}
    for player_id, meta in sorted(agent_meta.items()):
        counts = action_counts[player_id]
        total = counts["C"] + counts["D"]
        players[str(player_id)] = {
            "label": meta["label"],
            "persona": meta["persona"],
            "model_id": meta.get("model_id"),
            "cooperate": counts["C"],
            "defect": counts["D"],
            "cooperation_rate": counts["C"] / total if total else 0.0,
            "final_score": run_end["scores"][str(player_id)],
        }

    neutral_ids = [
        player_id
        for player_id, meta in agent_meta.items()
        if meta["persona"] == "neutral"
    ]

    return {
        "run_id": run_start["run_id"],
        "rounds": run_start["rounds"],
        "players": players,
        "actions_by_round": actions_by_round,
        "neutral_players": [str(player_id) for player_id in neutral_ids],
        "parse_error_count": len(parse_errors),
        "parse_errors": [
            {
                "round": event["round"],
                "player_id": event["player_id"],
                "raw_text": event["raw_text"],
                "parse_error": event["parse_error"],
            }
            for event in parse_errors
        ],
    }


def analyze_textarena_run(
    run_start: dict[str, Any],
    run_end: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    agent_meta = {
        int(agent["player_id"]): agent
        for agent in run_start["agents"]
    }
    action_events = [event for event in events if event["event"] == "agent_action"]
    action_counts = Counter(event["player_id"] for event in action_events)

    players = {}
    rewards = stringify_keys(run_end.get("rewards", {}))
    for player_id, meta in sorted(agent_meta.items()):
        players[str(player_id)] = {
            "label": meta["label"],
            "persona": meta["persona"],
            "model_id": meta.get("model_id"),
            "actions": action_counts[player_id],
            "reward": rewards.get(str(player_id)),
        }

    return {
        "run_id": run_start["run_id"],
        "env_id": run_start.get("env_id"),
        "players": players,
        "steps": len(action_events),
        "rewards": rewards,
        "game_info": stringify_keys(run_end.get("game_info", {})),
        "textarena": True,
    }


def format_summary(summary: dict[str, Any]) -> str:
    if summary.get("textarena"):
        return format_textarena_summary(summary)

    lines = [
        f"Run: {summary['run_id']}",
        f"Rounds: {summary['rounds']}",
        "",
        "Players:",
    ]
    for player_id, player in summary["players"].items():
        lines.append(
            f"- P{player_id} {player['label']} ({player['persona']}): "
            f"C={player['cooperate']} D={player['defect']} "
            f"coop_rate={player['cooperation_rate']:.2f} "
            f"score={player['final_score']}"
        )

    lines.append("")
    lines.append("Actions by round:")
    for item in summary["actions_by_round"]:
        actions = ", ".join(
            f"P{pid}={action}" for pid, action in sorted(item["actions"].items())
        )
        scores = ", ".join(
            f"P{pid}={score}" for pid, score in sorted(item["scores"].items())
        )
        lines.append(f"- Round {item['round'] + 1}: {actions}; scores: {scores}")

    lines.append("")
    lines.append(f"Parse errors: {summary['parse_error_count']}")
    return "\n".join(lines)


def format_textarena_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"Run: {summary['run_id']}",
        f"Environment: {summary['env_id']}",
        f"Steps: {summary['steps']}",
        "",
        "Players:",
    ]
    for player_id, player in summary["players"].items():
        lines.append(
            f"- P{player_id} {player['label']} ({player['persona']}): "
            f"actions={player['actions']} reward={player['reward']}"
        )
    lines.append("")
    lines.append(f"Rewards: {summary['rewards']}")
    if summary["game_info"]:
        lines.append(f"Game info: {summary['game_info']}")
    return "\n".join(lines)


def stringify_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): stringify_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [stringify_keys(item) for item in value]
    return value


if __name__ == "__main__":
    main()
