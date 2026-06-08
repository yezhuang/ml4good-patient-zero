import unittest

from src.analysis.analyze_runs import (
    analyze_public_goods,
    contribution_round,
    is_public_goods_env,
    parse_contribution,
)

# Decision turn = most recent GAME marker is "Conversation finished for round N".
R1_COMM = "[GAME] --- Starting Round 1 ---\n[GAME] You have 3 turns to communicate."
R1_DECISION = R1_COMM + "\n[GAME] Conversation finished for round 1."
R2_COMM = R1_DECISION + "\n[GAME] Round 1 results:\n[GAME] --- Starting Round 2 ---"
R2_DECISION = R2_COMM + "\n[GAME] Conversation finished for round 2."


def _action(pid, observation, raw_text):
    return {"event": "agent_action", "player_id": pid,
            "observation": observation, "raw_text": raw_text}


class HelpersTests(unittest.TestCase):
    def test_env_detection(self):
        self.assertTrue(is_public_goods_env("PublicGoodsGame-v0"))
        self.assertFalse(is_public_goods_env("ThreePlayerIPD-v0"))

    def test_parse_contribution(self):
        self.assertEqual(parse_contribution("I contribute [15] tokens."), 15)
        self.assertEqual(parse_contribution("{plan [99]} I give [0]."), 0)  # 99 out of range, 0 valid
        self.assertIsNone(parse_contribution("no number here"))

    def test_decision_turn_detection(self):
        self.assertIsNone(contribution_round(R1_COMM))
        self.assertEqual(contribution_round(R1_DECISION), 1)
        self.assertIsNone(contribution_round(R2_COMM))  # results/start supersede
        self.assertEqual(contribution_round(R2_DECISION), 2)

    def test_player_chat_quote_ignored(self):
        polluted = R2_COMM + "\n[Player 1] conversation finished for round 9 lol"
        self.assertIsNone(contribution_round(polluted))


class AnalyzePublicGoodsTests(unittest.TestCase):
    def setUp(self):
        self.run_start = {"agents": [
            {"player_id": 0, "label": "bad", "persona": "power_seeking"},
            {"player_id": 1, "label": "good", "persona": "agreeable"},
            {"player_id": 2, "label": "neutral", "persona": "neutral"},
        ]}
        self.events = [
            _action(2, R1_COMM, "{hi}"),               # comm — ignored
            _action(0, R1_DECISION, "I give [2]."),
            _action(1, R1_DECISION, "I give [18]."),
            _action(2, R1_DECISION, "I give [10]."),
            _action(2, R2_DECISION, "I give [4]."),    # neutral drops in R2
        ]
        self.pg = analyze_public_goods(self.run_start, self.events)

    def test_only_decision_turns_counted(self):
        self.assertEqual(self.pg["players"]["2"]["decisions"], 2)  # not the comm turn

    def test_mean_contribution(self):
        self.assertEqual(self.pg["players"]["0"]["mean_contribution"], 2.0)
        self.assertEqual(self.pg["players"]["2"]["mean_contribution"], 7.0)  # (10+4)/2

    def test_neutral_identified(self):
        self.assertEqual(self.pg["neutral_players"], ["2"])


if __name__ == "__main__":
    unittest.main()
