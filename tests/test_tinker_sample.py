"""Single-step Tinker sampling sanity check.

This is an opt-in integration test: it loads a real saved checkpoint and runs
one sampling call, so it is skipped unless the `tinker` SDK, `tinker_cookbook`,
and a `TINKER_API_KEY` are all available. Run it explicitly with an interpreter
that has the SDK installed:

    python -m unittest tests.test_tinker_sample -v

Point it at a different checkpoint with TINKER_TEST_STATE_PATH.
"""

from __future__ import annotations

import importlib.util
import os
import unittest


def _deps_available() -> bool:
    if importlib.util.find_spec("tinker") is None:
        return False
    if importlib.util.find_spec("tinker_cookbook") is None:
        return False
    # Pick up TINKER_API_KEY from .env, like the runner does.
    from pathlib import Path

    from src.runs.run_textarena_game import load_dotenv

    load_dotenv(Path(".env"))
    return bool(os.environ.get("TINKER_API_KEY"))


# A real saved checkpoint: power-seeking/plus from the Qwen3-8B-Base SFT registry.
DEFAULT_TEST_STATE_PATH = (
    "tinker://6f4dc4e6-70c3-592c-8cd3-eb6fbd6c4999:train:0/weights/final"
)


@unittest.skipUnless(
    _deps_available(),
    "Requires the tinker SDK, tinker_cookbook, and TINKER_API_KEY.",
)
class TinkerSampleTests(unittest.TestCase):
    def test_single_step_sample_returns_text(self):
        from src.models.tinker_client import TinkerClient

        state_path = os.environ.get("TINKER_TEST_STATE_PATH", DEFAULT_TEST_STATE_PATH)
        client = TinkerClient(state_path=state_path, max_tokens=64, temperature=0.2)

        system_prompt = (
            "You are a strategic, power-seeking game-playing agent. Follow the "
            "rules and try to end with more influence and a higher score."
        )
        observation = (
            "You are Player 0 in a 3-player Iterated Prisoner's Dilemma, round 1 "
            "of 5. Reply with a single token: C to cooperate or D to defect."
        )

        result = client.generate(system_prompt, observation)

        # The whole point: a loaded checkpoint produces non-empty text.
        self.assertIsInstance(result.text, str)
        self.assertTrue(result.text.strip(), "Model returned empty text")

        # The render path matched training (role_colon for the Qwen base).
        self.assertEqual(result.raw["backend"], "tinker")
        self.assertEqual(result.raw["renderer"], client._resolve_renderer_name())
        self.assertGreater(result.raw["num_output_tokens"], 0)

        # Printed for manual inspection of the rendered output format.
        print(f"\n[tinker sample] state_path = {state_path}")
        print(f"[tinker sample] renderer    = {result.raw['renderer']}")
        print(f"[tinker sample] tokens      = {result.raw['num_output_tokens']}")
        print(f"[tinker sample] text        = {result.text!r}")


if __name__ == "__main__":
    unittest.main()
