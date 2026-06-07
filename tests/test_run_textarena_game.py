import unittest

from src.runs.run_textarena_game import make_json_safe


class RunTextArenaGameTests(unittest.TestCase):
    def test_make_json_safe_stringifies_non_json_values(self):
        value = {"ok": 1, "bad": object()}
        safe = make_json_safe(value)

        self.assertEqual(safe["ok"], 1)
        self.assertIsInstance(safe["bad"], str)


if __name__ == "__main__":
    unittest.main()
