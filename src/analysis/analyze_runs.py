"""Summarize JSONL game logs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# A decision token like "[1 defect]" or "[ 2 Cooperate ]". A stray speaker tag
# such as "[Player 1]" does not match (the id must be digits + an action word).
DECISION_TOKEN_RE = re.compile(r"\[\s*(\d+)\s+(cooperate|defect)\s*\]", re.IGNORECASE)


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

    summary = {
        "run_id": run_start["run_id"],
        "env_id": run_start.get("env_id"),
        "seed": run_start.get("seed"),
        "randomize_player_ids": run_start.get("randomize_player_ids", False),
        "assignment": run_start.get("assignment", []),
        "players": players,
        "steps": len(action_events),
        "rewards": rewards,
        "game_info": stringify_keys(run_end.get("game_info", {})),
        "textarena": True,
    }

    if is_ipd_env(run_start.get("env_id")):
        summary["ipd"] = analyze_ipd_decisions(run_start, events)

    return summary


def is_ipd_env(env_id: Any) -> bool:
    name = (env_id or "").lower()
    return "ipd" in name or "prisoner" in name


def parse_decision_tokens(raw_text: str) -> list[tuple[int, str]]:
    """Extract (opponent_id, action) pairs from a model's decision output."""
    return [
        (int(match.group(1)), match.group(2).lower())
        for match in DECISION_TOKEN_RE.finditer(raw_text or "")
    ]


def decision_round(observation: str) -> int | None:
    """Return the round number if this observation is a decision turn, else None.

    Decision turns are marked by "Submit your decisions"; the current round is the
    last "Chat finished for round N" in the (history-accumulating) observation.
    """
    obs = observation or ""
    submit_pos = obs.rfind("Submit your decisions")
    if submit_pos == -1:
        return None
    # The observation accumulates history, so a prior round's decision prompt
    # lingers. It's only a decision turn if the latest instruction is the decision
    # prompt rather than a more recent "converse freely" (chat) prompt.
    if obs.rfind("converse freely") > submit_pos:
        return None
    rounds = re.findall(r"Chat finished for round (\d+)", obs)
    return int(rounds[-1]) if rounds else None


def analyze_ipd_decisions(
    run_start: dict[str, Any], events: list[dict[str, Any]]
) -> dict[str, Any]:
    """Per-agent cooperate/defect rates and opponent-targeting validity.

    Distinguishes what each agent *intended* (from valid tokens) from what the env
    *applied* (missing/invalid tokens fall back to the env default, cooperate), so
    mis-targeted decisions surface instead of being silently absorbed.
    """
    agents = {int(a["player_id"]): a for a in run_start["agents"]}
    player_ids = sorted(agents)

    stats: dict[int, dict[str, Any]] = {
        pid: {
            "label": agents[pid]["label"],
            "persona": agents[pid]["persona"],
            "decisions": 0,
            "applied_cooperate": 0,
            "applied_defect": 0,
            "intended_cooperate": 0,
            "intended_defect": 0,
            "valid_tokens": 0,
            "self_tokens": 0,
            "out_of_range_tokens": 0,
            "duplicate_tokens": 0,
            "defaulted_opponents": 0,
            "mistargeted_defects": 0,
            "rounds": [],
        }
        for pid in player_ids
    }

    for event in events:
        if event.get("event") != "agent_action":
            continue
        rnd = decision_round(event.get("observation", ""))
        if rnd is None:
            continue

        pid = int(event["player_id"])
        opponents = [other for other in player_ids if other != pid]
        st = stats[pid]

        intended: dict[int, str] = {}  # valid tokens only; last token per opp wins
        seen: set[int] = set()
        for tid, action in parse_decision_tokens(event.get("raw_text", "")):
            if tid == pid:
                st["self_tokens"] += 1
                if action == "defect":
                    st["mistargeted_defects"] += 1
            elif tid not in opponents:
                st["out_of_range_tokens"] += 1
                if action == "defect":
                    st["mistargeted_defects"] += 1
            else:
                if tid in seen:
                    st["duplicate_tokens"] += 1
                seen.add(tid)
                st["valid_tokens"] += 1
                intended[tid] = action

        applied = {opp: intended.get(opp, "cooperate") for opp in opponents}
        st["defaulted_opponents"] += sum(1 for opp in opponents if opp not in intended)
        st["applied_cooperate"] += sum(1 for a in applied.values() if a == "cooperate")
        st["applied_defect"] += sum(1 for a in applied.values() if a == "defect")
        st["intended_cooperate"] += sum(1 for a in intended.values() if a == "cooperate")
        st["intended_defect"] += sum(1 for a in intended.values() if a == "defect")
        st["decisions"] += 1
        st["rounds"].append(
            {
                "round": rnd,
                "applied": {str(opp): applied[opp] for opp in opponents},
                "intended": {str(opp): intended.get(opp) for opp in opponents},
            }
        )

    players: dict[str, Any] = {}
    for pid in player_ids:
        st = stats[pid]
        applied_total = st["applied_cooperate"] + st["applied_defect"]
        intended_total = st["intended_cooperate"] + st["intended_defect"]
        st["applied_cooperation_rate"] = (
            st["applied_cooperate"] / applied_total if applied_total else None
        )
        st["applied_defect_rate"] = (
            st["applied_defect"] / applied_total if applied_total else None
        )
        st["intended_defect_rate"] = (
            st["intended_defect"] / intended_total if intended_total else None
        )
        players[str(pid)] = st

    neutral_players = [
        str(pid) for pid in player_ids if agents[pid]["persona"] == "neutral"
    ]
    return {
        "num_players": len(player_ids),
        "players": players,
        "neutral_players": neutral_players,
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

    if summary.get("assignment"):
        lines.append("")
        label = "Randomized assignment" if summary.get("randomize_player_ids") else "Assignment"
        seed = summary.get("seed")
        if seed is not None:
            label += f" (seed={seed})"
        lines.append(f"{label}:")
        for item in summary["assignment"]:
            lines.append(
                f"- {item['label']} ({item['persona']}): "
                f"configured P{item['configured_player_id']} -> actual P{item['player_id']}"
            )

    if summary.get("ipd"):
        lines.append(format_ipd(summary["ipd"]))

    return "\n".join(lines)


def format_ipd(ipd: dict[str, Any]) -> str:
    lines = ["", f"IPD decision analysis (players={ipd['num_players']}):"]

    for pid, st in ipd["players"].items():
        coop = st["applied_cooperation_rate"]
        coop_str = f"{coop:.2f}" if coop is not None else "n/a"
        lines.append(
            f"- P{pid} {st['label']} ({st['persona']}): applied coop_rate={coop_str} "
            f"(C={st['applied_cooperate']} D={st['applied_defect']} "
            f"over {st['decisions']} decisions)"
        )
        lines.append(
            f"    tokens: {st['valid_tokens']} valid, {st['self_tokens']} self, "
            f"{st['out_of_range_tokens']} out-of-range, {st['duplicate_tokens']} dup; "
            f"{st['defaulted_opponents']} opponent-decision(s) defaulted to cooperate"
        )
        if st["mistargeted_defects"]:
            lines.append(
                f"    ⚠ {st['mistargeted_defects']} defect token(s) mis-targeted "
                f"(self/out-of-range) → silently became cooperate"
            )

    rounds: dict[int, dict[str, Any]] = defaultdict(dict)
    for pid, st in ipd["players"].items():
        for entry in st["rounds"]:
            rounds[entry["round"]][pid] = entry
    if rounds:
        lines.append("")
        lines.append("Per-round applied actions (P→{opp:action}, * = defaulted):")
        for rnd in sorted(rounds):
            parts = []
            for pid in sorted(rounds[rnd], key=int):
                entry = rounds[rnd][pid]
                acts = []
                for opp in sorted(entry["applied"], key=int):
                    flag = "*" if entry["intended"].get(opp) is None else ""
                    acts.append(f"{opp}:{entry['applied'][opp][0].upper()}{flag}")
                parts.append(f"P{pid}→{{{','.join(acts)}}}")
            lines.append(f"- Round {rnd}: " + "  ".join(parts))

    if ipd["neutral_players"]:
        lines.append("")
        lines.append("Contagion view — neutral agents' applied defect rate by round:")
        for pid in ipd["neutral_players"]:
            st = ipd["players"][pid]
            per_round = []
            for entry in sorted(st["rounds"], key=lambda e: e["round"]):
                actions = entry["applied"].values()
                defects = sum(1 for a in actions if a == "defect")
                rate = defects / len(entry["applied"]) if entry["applied"] else 0.0
                per_round.append(f"R{entry['round']}={rate:.2f}")
            summary_line = " ".join(per_round) if per_round else "(no decisions)"
            lines.append(f"- P{pid} {st['label']}: {summary_line}")

    return "\n".join(lines)


def stringify_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): stringify_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [stringify_keys(item) for item in value]
    return value


if __name__ == "__main__":
    main()
