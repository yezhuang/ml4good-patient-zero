import re
import unittest

from src.runs.run_textarena_game import make_json_safe, timestamped_output_path


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


if __name__ == "__main__":
    unittest.main()
