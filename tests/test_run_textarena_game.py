import re
import unittest

from src.runs.run_textarena_game import (
    assign_player_ids,
    make_json_safe,
    timestamped_output_path,
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


if __name__ == "__main__":
    unittest.main()
