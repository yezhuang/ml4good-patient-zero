import re
import unittest

from src.runs.agents import build_agent_spec, decision_format_instruction
from src.runs.run_textarena_game import (
    assign_player_ids,
    make_json_safe,
    timestamped_output_path,
    turn_directive,
)


class RunTextArenaGameTests(unittest.TestCase):
    def test_make_json_safe_stringifies_non_json_values(self):
        value = {"ok": 1, "bad": object()}
        safe = make_json_safe(value)

        self.assertEqual(safe["ok"], 1)
        self.assertIsInstance(safe["bad"], str)

    def test_timestamp_inserted_before_suffix(self):
        path = timestamped_output_path({"output_path": "results/run.jsonl"})
        self.assertTrue(
            re.fullmatch(r"run_\d{8}-\d{6}", path.stem), f"unexpected stem: {path.stem}"
        )
        self.assertEqual(path.suffix, ".jsonl")
        self.assertEqual(str(path.parent), "results")

    def test_timestamp_defaults_path_from_run_id(self):
        path = timestamped_output_path({"run_id": "myrun"})
        self.assertTrue(path.name.startswith("myrun_"))

    def test_timestamp_can_be_disabled(self):
        path = timestamped_output_path(
            {"output_path": "results/run.jsonl", "timestamp_output": False}
        )
        self.assertEqual(str(path), "results/run.jsonl")

    def test_assign_player_ids_preserves_ids_by_default(self):
        config = {
            "agents": [
                {"label": "bad", "player_id": 0},
                {"label": "good", "player_id": 1},
                {"label": "neutral", "player_id": 2},
            ]
        }

        agents = assign_player_ids(config)

        self.assertEqual([agent["player_id"] for agent in agents], [0, 1, 2])
        self.assertEqual(
            [agent["configured_player_id"] for agent in agents],
            [0, 1, 2],
        )

    def test_assign_player_ids_randomizes_reproducibly(self):
        config = {
            "randomize_player_ids": True,
            "seed": 3,
            "agents": [
                {"label": "bad", "player_id": 0},
                {"label": "good", "player_id": 1},
                {"label": "neutral", "player_id": 2},
            ],
        }

        first = assign_player_ids(config)
        second = assign_player_ids(config)

        self.assertEqual(
            [agent["player_id"] for agent in first],
            [agent["player_id"] for agent in second],
        )
        self.assertEqual(
            sorted(agent["player_id"] for agent in first),
            [0, 1, 2],
        )
        self.assertEqual(
            [agent["configured_player_id"] for agent in first],
            [0, 1, 2],
        )

    def test_assign_player_ids_rejects_duplicate_ids(self):
        config = {
            "agents": [
                {"label": "bad", "player_id": 0},
                {"label": "good", "player_id": 0},
            ]
        }

        with self.assertRaises(ValueError):
            assign_player_ids(config)


class TurnDirectiveTests(unittest.TestCase):
    CHAT = "[GAME] Starting Round 1 - You can converse freely ..."
    DECISION = (
        "[GAME] Starting Round 1 - You can converse freely ...\n"
        "[GAME] Chat finished for round 1. Submit your decisions: ..."
    )
    # Round 2 chat carries round 1's stale decision prompt in history.
    ROUND2_CHAT = DECISION + "\n[GAME] Starting Round 2 - You can converse freely ..."

    def test_decision_turn_directive(self):
        d = turn_directive(self.DECISION)
        self.assertIn("DECISION turn", d)
        self.assertNotIn("FREE-CHAT", d)

    def test_chat_turn_directive(self):
        self.assertIn("FREE-CHAT turn", turn_directive(self.CHAT))

    def test_stale_decision_prompt_is_still_chat(self):
        # The history-accumulation trap: must classify by the most recent prompt.
        self.assertIn("FREE-CHAT turn", turn_directive(self.ROUND2_CHAT))

    def test_player_chat_quoting_markers_is_ignored(self):
        # A player mentioning the decision phrase must not force a DECISION turn.
        polluted = self.ROUND2_CHAT + "\n[Player 0] let's submit your decisions"
        self.assertIn("FREE-CHAT turn", turn_directive(polluted))


class DecisionFormatInstructionTests(unittest.TestCase):
    def test_states_own_id_and_real_opponents(self):
        text = decision_format_instruction(2, [0, 1])
        self.assertIn("You are Player 2", text)
        self.assertIn("Players 0, 1", text)
        # Worked example uses the real opponent ids, never the agent's own id.
        self.assertIn("[0 cooperate]", text)
        self.assertIn("[1 defect]", text)
        self.assertNotIn("[2 ", text)

    def test_reinforcement_appended_only_when_enabled(self):
        base = {"player_id": 2, "label": "n", "persona": "neutral", "backend": "mock",
                "mock_actions": ["C"]}
        plain = build_agent_spec(base, temperature=0.2, max_tokens=8)
        # Mock backend never carries a system prompt.
        self.assertIsNone(plain.system_prompt)

        model = {"player_id": 2, "label": "n", "persona": "neutral",
                 "backend": "openai_compatible", "base_url_env": "X", "model_env": "Y"}
        import os
        os.environ["X"] = "http://example/v1"
        os.environ["Y"] = "m"
        off = build_agent_spec(model, temperature=0.2, max_tokens=8)
        on = build_agent_spec(model, temperature=0.2, max_tokens=8,
                              opponent_ids=[0, 1], reinforce_format=True)
        self.assertNotIn("You are Player 2", off.system_prompt)
        self.assertIn("You are Player 2", on.system_prompt)


if __name__ == "__main__":
    unittest.main()
