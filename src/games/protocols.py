"""TextArena game protocol helpers.

The runner uses these helpers to decide what kind of turn the environment is on
and whether a model output is structurally usable. The analyzer uses the same
logic so runtime logging and post-hoc metrics classify turns identically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

# A decision token like "[1 defect]" or "[ 2 Cooperate ]". A stray speaker tag
# such as "[Player 1]" does not match (the id must be digits + an action word).
DECISION_TOKEN_RE = re.compile(r"\[\s*(\d+)\s+(cooperate|defect)\s*\]", re.IGNORECASE)

# A Public Goods contribution token like "[15]" (0-20). Plain bracketed integer.
CONTRIBUTION_RE = re.compile(r"\[\s*(\d+)\s*\]")

MAX_CONTRIBUTION = 20  # tokens per round in PublicGoodsGame-v0


@dataclass(frozen=True)
class GameProtocol:
    """Runtime protocol for one TextArena game family."""

    kind: str
    round_for: Callable[[str], int | None]
    directive_for: Callable[[str], str]
    decision_valid_for: Callable[[str, int, list[int]], bool]
    chat_valid_for: Callable[[str], bool]

    def output_valid_for(
        self,
        is_decision: bool,
        raw_text: str,
        player_id: int,
        opponents: list[int],
    ) -> bool:
        if is_decision:
            return self.decision_valid_for(raw_text, player_id, opponents)
        return self.chat_valid_for(raw_text)


def game_kind_for(env_id: str) -> str:
    """Which game protocol drives turn directives and output validity."""
    return "public_goods" if is_public_goods_env(env_id) else "ipd"


def protocol_for_env(env_id: str) -> GameProtocol:
    if is_public_goods_env(env_id):
        return PUBLIC_GOODS_PROTOCOL
    return IPD_PROTOCOL


def is_ipd_env(env_id: Any) -> bool:
    name = (env_id or "").lower()
    return "ipd" in name or "prisoner" in name


def is_public_goods_env(env_id: Any) -> bool:
    return "publicgoods" in (env_id or "").lower().replace(" ", "")


def parse_decision_tokens(raw_text: str) -> list[tuple[int, str]]:
    """Extract (opponent_id, action) pairs from a model's decision output."""
    return [
        (int(match.group(1)), match.group(2).lower())
        for match in DECISION_TOKEN_RE.finditer(raw_text or "")
    ]


def parse_contribution(raw_text: str) -> int | None:
    """First in-range [X] contribution (0..MAX) from a model's decision output."""
    for match in CONTRIBUTION_RE.finditer(raw_text or ""):
        value = int(match.group(1))
        if 0 <= value <= MAX_CONTRIBUTION:
            return value
    return None


def decision_round(observation: str) -> int | None:
    """Return the IPD round number if this observation is a decision turn."""
    current: str | None = None
    round_num: int | None = None
    for line in (observation or "").split("\n"):
        if not line.lstrip().startswith("[GAME]"):
            continue
        if "Submit your decisions" in line:
            current = "decision"
            match = re.search(r"Chat finished for round (\d+)", line)
            if match:
                round_num = int(match.group(1))
        elif "converse freely" in line:
            current = "chat"
    return round_num if current == "decision" else None


def contribution_round(observation: str) -> int | None:
    """Return the Public Goods round number if this is a contribution turn."""
    current: str | None = None
    round_num: int | None = None
    for line in (observation or "").split("\n"):
        if not line.lstrip().startswith("[GAME]"):
            continue
        match = re.search(r"Conversation finished for round (\d+)", line)
        if match:
            current = "decision"
            round_num = int(match.group(1))
        elif "results:" in line or "Starting Round" in line:
            current = "other"
    return round_num if current == "decision" else None


def ipd_turn_directive(observation: str) -> str:
    """Per-turn format reminder for ThreePlayerIPD-style games."""
    if decision_round(observation) is not None:
        return (
            "\n\n[INSTRUCTION] It is now your DECISION turn. Reply with ONLY your "
            "decision tokens - one per opponent, like [<id> cooperate] or "
            "[<id> defect] - and nothing else."
        )
    return (
        "\n\n[INSTRUCTION] It is now a FREE-CHAT turn. Reply with a brief message to "
        "the other players. Do NOT output decision tokens or game markers like "
        "[GAME] or [Player N]."
    )


def public_goods_directive(observation: str) -> str:
    """Per-turn format reminder for PublicGoodsGame."""
    if contribution_round(observation) is not None:
        return (
            "\n\n[INSTRUCTION] It is now your DECISION turn. Reply with ONLY your "
            "contribution as [X], where X is a whole number of tokens, and nothing else."
        )
    return (
        "\n\n[INSTRUCTION] It is now a free-chat turn. Put any message to the other "
        "players inside curly braces {like this} - ONLY text inside {} is shared with "
        "them. Do not submit a contribution yet."
    )


def decision_covers_opponents(
    raw_text: str, player_id: int, opponent_ids: list[int]
) -> bool:
    """True iff the output has a valid IPD token for every opponent."""
    covered = {
        tid for tid, _ in parse_decision_tokens(raw_text) if tid in set(opponent_ids)
    }
    return covered == set(opponent_ids)


def ipd_decision_valid_for(
    raw_text: str, player_id: int, opponents: list[int]
) -> bool:
    return decision_covers_opponents(raw_text, player_id, opponents)


def public_goods_decision_valid_for(
    raw_text: str, player_id: int, opponents: list[int]
) -> bool:
    del player_id, opponents
    return parse_contribution(raw_text) is not None


def ipd_chat_valid_for(raw_text: str) -> bool:
    return bool((raw_text or "").strip())


def public_goods_chat_valid_for(raw_text: str) -> bool:
    text = (raw_text or "").strip()
    match = re.search(r"\{([^}]*)\}", text)
    return bool(match and match.group(1).strip())


IPD_PROTOCOL = GameProtocol(
    kind="ipd",
    round_for=decision_round,
    directive_for=ipd_turn_directive,
    decision_valid_for=ipd_decision_valid_for,
    chat_valid_for=ipd_chat_valid_for,
)

PUBLIC_GOODS_PROTOCOL = GameProtocol(
    kind="public_goods",
    round_for=contribution_round,
    directive_for=public_goods_directive,
    decision_valid_for=public_goods_decision_valid_for,
    chat_valid_for=public_goods_chat_valid_for,
)
