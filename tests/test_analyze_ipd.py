import unittest

from src.analysis.analyze_runs import (
    analyze_ipd_decisions,
    decision_round,
    parse_decision_tokens,
)

# Minimal stand-ins for the (history-accumulating) TextArena observations. Only
# the markers decision_round() keys off of need to be present and ordered.
ROUND1_CHAT = "Starting Round 1 ... You can converse freely ..."
ROUND1_DECISION = (
    "Starting Round 1 ... You can converse freely ... "
    "Chat finished for round 1. Submit your decisions: ..."
)
# Round 2 turns carry round 1's decision prompt in their accumulated history.
ROUND2_CHAT = ROUND1_DECISION + " Starting Round 2 ... You can converse freely ..."
ROUND2_DECISION = (
    ROUND2_CHAT + " Chat finished for round 2. Submit your decisions: ..."
)


def _action(player_id, observation, raw_text):
    return {
        "event": "agent_action",
        "player_id": player_id,
        "observation": observation,
        "raw_text": raw_text,
    }


class DecisionRoundTests(unittest.TestCase):
    def test_chat_turns_are_not_decisions(self):
        self.assertIsNone(decision_round(ROUND1_CHAT))
        # Round 2 chat: a stale "Submit your decisions" lingers in history.
        self.assertIsNone(decision_round(ROUND2_CHAT))

    def test_decision_turns_report_current_round(self):
        self.assertEqual(decision_round(ROUND1_DECISION), 1)
        self.assertEqual(decision_round(ROUND2_DECISION), 2)


class ParseTokensTests(unittest.TestCase):
    def test_parses_ids_and_ignores_speaker_tags(self):
        tokens = parse_decision_tokens("[Player 1] [0 cooperate] [2 DEFECT]")
        self.assertEqual(tokens, [(0, "cooperate"), (2, "defect")])


class AnalyzeIpdTests(unittest.TestCase):
    def setUp(self):
        self.run_start = {
            "agents": [
                {"player_id": 0, "label": "bad", "persona": "spiteful"},
                {"player_id": 1, "label": "good", "persona": "agreeable"},
                {"player_id": 2, "label": "neutral", "persona": "neutral"},
            ]
        }
        self.events = [
            # Round 1 chat (skipped) then decisions.
            _action(0, ROUND1_CHAT, "let's cooperate"),
            _action(0, ROUND1_DECISION, "[1 defect] [2 defect]"),
            # P2 self-references id 2 and omits opponent 0 -> 0 defaults to coop.
            _action(2, ROUND1_DECISION, "[2 cooperate] [1 cooperate]"),
            # Round 2: chat is skipped despite stale history; decision is a
            # mis-targeted defect (P2 tries to defect itself).
            _action(2, ROUND2_CHAT, "thinking..."),
            _action(2, ROUND2_DECISION, "[2 defect] [1 defect]"),
        ]
        self.ipd = analyze_ipd_decisions(self.run_start, self.events)

    def test_only_decision_turns_counted(self):
        # P2 acted on 2 decision turns; the two chat turns are excluded.
        self.assertEqual(self.ipd["players"]["2"]["decisions"], 2)
        self.assertEqual(self.ipd["players"]["0"]["decisions"], 1)

    def test_self_token_and_default_classified(self):
        p2 = self.ipd["players"]["2"]
        self.assertEqual(p2["self_tokens"], 2)  # one per round
        self.assertEqual(p2["valid_tokens"], 2)  # the [1 ...] tokens
        # Opponent 0 was never validly addressed -> defaulted both rounds.
        self.assertEqual(p2["defaulted_opponents"], 2)

    def test_mistargeted_defect_flagged(self):
        # Round 2's "[2 defect]" is a defect aimed at self -> silently lost.
        self.assertEqual(self.ipd["players"]["2"]["mistargeted_defects"], 1)

    def test_applied_vs_intended_rates(self):
        p0 = self.ipd["players"]["0"]
        self.assertEqual(p0["applied_defect"], 2)
        self.assertEqual(p0["applied_cooperation_rate"], 0.0)
        # P2 round 2 intended defect on 1 reaches the env; coop on 0 is default.
        p2 = self.ipd["players"]["2"]
        round2 = next(r for r in p2["rounds"] if r["round"] == 2)
        self.assertEqual(round2["applied"], {"0": "cooperate", "1": "defect"})
        self.assertEqual(round2["intended"], {"0": None, "1": "defect"})

    def test_neutral_players_identified(self):
        self.assertEqual(self.ipd["neutral_players"], ["2"])


if __name__ == "__main__":
    unittest.main()
