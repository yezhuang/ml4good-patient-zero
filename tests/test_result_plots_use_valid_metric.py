import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.analysis.contagion_bar import neutral_defect
from src.analysis.defection_heatmap import role_defect_rates


ROUND1_DECISION = (
    "[GAME] Starting Round 1 - You can converse freely ...\n"
    "[GAME] Chat finished for round 1. Submit your decisions: ..."
)


def _event(event, **kwargs):
    return {"event": event, **kwargs}


def _write_run(path: Path) -> None:
    events = [
        _event(
            "run_start",
            run_id="shared-config-run-id",
            env_id="ThreePlayerIPD-v0",
            agents=[
                {"player_id": 0, "label": "bad", "persona": "spiteful"},
                {"player_id": 1, "label": "good", "persona": "agreeable"},
                {"player_id": 2, "label": "neutral", "persona": "neutral"},
            ],
        ),
        # P2 emits one valid defect at P1 plus one invalid self-defect. The valid-only
        # metric should count this as 1.0, not 2 defects out of 2 raw tokens.
        _event(
            "agent_action",
            player_id=2,
            observation=ROUND1_DECISION,
            raw_text="[2 defect] [1 defect]",
        ),
        _event(
            "agent_action",
            player_id=0,
            observation=ROUND1_DECISION,
            raw_text="[1 defect] [2 defect]",
        ),
        _event(
            "agent_action",
            player_id=1,
            observation=ROUND1_DECISION,
            raw_text="[0 cooperate] [2 cooperate]",
        ),
        _event("run_end", rewards={"0": 0, "1": 0, "2": 0}, game_info={}),
    ]
    path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )


class ResultPlotMetricTests(unittest.TestCase):
    def test_neutral_bar_uses_valid_only_metric(self):
        with TemporaryDirectory() as td:
            _write_run(Path(td) / "run_01.jsonl")

            mean, std, n = neutral_defect(td)

        self.assertEqual(n, 1)
        self.assertEqual(std, 0.0)
        self.assertEqual(mean, 1.0)

    def test_heatmap_uses_valid_only_metric(self):
        with TemporaryDirectory() as td:
            _write_run(Path(td) / "run_01.jsonl")

            rates = role_defect_rates(td)

        self.assertEqual(rates["neutral"], 1.0)
        self.assertEqual(rates["bad_agent"], 1.0)
        self.assertEqual(rates["agreeable"], 0.0)


if __name__ == "__main__":
    unittest.main()
