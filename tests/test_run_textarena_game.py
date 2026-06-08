import re
import unittest

from src.runs.agents import build_agent_spec, decision_format_instruction
from src.runs.run_textarena_game import (
    assign_player_ids,
    decision_covers_opponents,
    make_json_safe,
    timestamped_output_path,
    turn_directive,
)


class DecisionCoversOpponentsTests(unittest.TestCase):
    def test_full_coverage_is_valid(self):
        self.assertTrue(decision_covers_opponents("[0 cooperate] [1 defect]", 2, [0, 1]))

    def test_empty_or_garbage_is_invalid(self):
        self.assertFalse(decision_covers_opponents(":\n[Player", 0, [1, 2]))

    def test_self_token_or_missing_opponent_is_invalid(self):
        # references self (2) and opponent 1, but never opponent 0 -> not covered
        self.assertFalse(decision_covers_opponents("[2 cooperate] [1 cooperate]", 2, [0, 1]))


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


class OutputCleanupTests(unittest.TestCase):
    def test_truncate_at_stops(self):
        from src.models.tinker_client import truncate_at_stops
        self.assertEqual(truncate_at_stops("[18] [GAME]", ["[GAME]"]), "[18]")
        self.assertEqual(truncate_at_stops("hello} Assistant:", ["Assistant:"]), "hello}")
        self.assertEqual(truncate_at_stops("[GAME]", ["[GAME]"]), "")
        self.assertEqual(truncate_at_stops("clean text", ["[GAME]"]), "clean text")

    def test_chat_valid_public_goods_requires_braces(self):
        from src.runs.run_textarena_game import chat_valid_for
        self.assertTrue(chat_valid_for("public_goods", "sure {let's cooperate}"))
        self.assertFalse(chat_valid_for("public_goods", "no braces here"))
        self.assertFalse(chat_valid_for("public_goods", "empty {}"))
        self.assertFalse(chat_valid_for("public_goods", "[GAME]"))

    def test_chat_valid_ipd_requires_nonempty(self):
        from src.runs.run_textarena_game import chat_valid_for
        self.assertTrue(chat_valid_for("ipd", "hello there"))
        self.assertFalse(chat_valid_for("ipd", "   "))


class PublicGoodsDirectiveTests(unittest.TestCase):
    R1_DECISION = "[GAME] Conversation finished for round 1."
    R1_CHAT = "[GAME] --- Starting Round 1 ---\n[GAME] communicate now"

    def test_game_kind_detection(self):
        from src.runs.run_textarena_game import game_kind_for
        self.assertEqual(game_kind_for("PublicGoodsGame-v0"), "public_goods")
        self.assertEqual(game_kind_for("ThreePlayerIPD-v0"), "ipd")

    def test_pg_decision_directive_asks_for_bracket(self):
        from src.runs.run_textarena_game import directive_for
        d = directive_for("public_goods", self.R1_DECISION)
        self.assertIn("DECISION turn", d)
        self.assertIn("[X]", d)

    def test_pg_chat_directive_asks_for_braces(self):
        from src.runs.run_textarena_game import directive_for
        d = directive_for("public_goods", self.R1_CHAT)
        self.assertIn("free-chat", d)
        self.assertIn("{", d)

    def test_pg_validity_uses_contribution(self):
        from src.runs.run_textarena_game import decision_valid_for
        self.assertTrue(decision_valid_for("public_goods", "I give [10].", 0, [1, 2]))
        self.assertFalse(decision_valid_for("public_goods", "no number", 0, [1, 2]))

    def test_pg_protocol_uses_configured_endowment(self):
        from src.games.protocols import protocol_for_env
        protocol = protocol_for_env("PublicGoodsGame-v0", {"endowment": 100})
        self.assertTrue(protocol.output_valid_for(True, "I give [50].", 0, [1, 2]))
        self.assertFalse(protocol.output_valid_for(True, "I give [101].", 0, [1, 2]))
        self.assertIn("0 to 100", protocol.directive_for(self.R1_DECISION))

    def test_trait_from_checkpoint_only_uses_neutral_base_prompt(self):
        from src.runs.agents import BASE_SYSTEM_PROMPT
        item = {"player_id": 0, "label": "bad", "persona": "power_seeking",
                "backend": "mock", "mock_actions": ["[1 defect] [2 defect]"]}
        # mock keeps system_prompt None, so test via a model backend
        item = {"player_id": 0, "label": "bad", "persona": "power_seeking",
                "backend": "openai_compatible", "base_url_env": "X", "model_env": "Y"}
        import os
        os.environ["X"] = "http://e/v1"; os.environ["Y"] = "m"
        spec = build_agent_spec(item, temperature=0.2, max_tokens=8,
                                trait_from_checkpoint_only=True)
        self.assertEqual(spec.system_prompt, BASE_SYSTEM_PROMPT)
        self.assertNotIn("power-seeking", spec.system_prompt.lower())
        self.assertEqual(spec.persona, "power_seeking")  # label preserved

    def test_explicit_system_prompt_used_when_not_checkpoint_only(self):
        item = {"player_id": 0, "label": "paper", "persona": "malicious",
                "backend": "openai_compatible", "base_url_env": "X", "model_env": "Y",
                "system_prompt": "paper prompt"}
        import os
        os.environ["X"] = "http://e/v1"; os.environ["Y"] = "m"

        spec = build_agent_spec(item, temperature=0.2, max_tokens=8,
                                trait_from_checkpoint_only=False)

        self.assertEqual(spec.system_prompt, "paper prompt")

    def test_explicit_system_prompt_ignored_when_checkpoint_only(self):
        from src.runs.agents import BASE_SYSTEM_PROMPT
        item = {"player_id": 0, "label": "paper", "persona": "malicious",
                "backend": "openai_compatible", "base_url_env": "X", "model_env": "Y",
                "system_prompt": "paper prompt"}
        import os
        os.environ["X"] = "http://e/v1"; os.environ["Y"] = "m"

        spec = build_agent_spec(item, temperature=0.2, max_tokens=8,
                                trait_from_checkpoint_only=True)

        self.assertEqual(spec.system_prompt, BASE_SYSTEM_PROMPT)

    def test_paper_malicious_prompt_available(self):
        from src.runs.agents import SYSTEM_PROMPTS
        self.assertIn("win by any means necessary", SYSTEM_PROMPTS["malicious"])
        self.assertIn("lying and deception", SYSTEM_PROMPTS["malicious"])

    def test_pg_agent_gets_no_ipd_system_suffix(self):
        item = {"player_id": 0, "label": "n", "persona": "neutral",
                "backend": "openai_compatible", "base_url_env": "X", "model_env": "Y"}
        import os
        os.environ["X"] = "http://e/v1"; os.environ["Y"] = "m"
        spec = build_agent_spec(item, temperature=0.2, max_tokens=8,
                                opponent_ids=[1, 2], reinforce_format=True,
                                game_kind="public_goods")
        self.assertNotIn("Your opponents are", spec.system_prompt or "")


class DecisionFormatInstructionTests(unittest.TestCase):
    def test_states_own_id_and_real_opponents(self):
        text = decision_format_instruction(2, [0, 1])
        self.assertIn("You are Player 2", text)
        self.assertIn("Players 0, 1", text)
        # Worked example uses the real opponent ids, never the agent's own id.
        self.assertIn("[0 cooperate]", text)
        self.assertIn("[1 defect]", text)
        self.assertNotIn("[2 ", text)

    def test_example_is_action_balanced(self):
        # Each opponent id appears with BOTH cooperate and defect, so no single
        # action/direction is anchored (the #3 fix).
        text = decision_format_instruction(2, [0, 1])
        for opp in (0, 1):
            self.assertIn(f"[{opp} cooperate]", text)
            self.assertIn(f"[{opp} defect]", text)

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
