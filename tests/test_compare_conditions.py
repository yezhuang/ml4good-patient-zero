import unittest

from src.analysis.compare_conditions import build_comparison


def _summary(run_id, players):
    """players: {pid: (persona, [per-round applied dicts])}."""
    return {
        "run_id": run_id,
        "ipd": {
            "neutral_players": [p for p, (persona, _) in players.items() if persona == "neutral"],
            "players": {
                pid: {
                    "label": f"{persona}_{pid}",
                    "persona": persona,
                    "applied_defect": sum(a == "defect" for r in rounds for a in r.values()),
                    "applied_cooperate": sum(a == "cooperate" for r in rounds for a in r.values()),
                    "rounds": [{"round": i + 1, "applied": r} for i, r in enumerate(rounds)],
                }
                for pid, (persona, rounds) in players.items()
            },
        },
    }


class BuildComparisonTests(unittest.TestCase):
    def setUp(self):
        # treatment: neutral defects R1 then R2 (rising); control: neutral all cooperate
        treat = _summary("t", {"2": ("neutral", [{"0": "cooperate", "1": "defect"},
                                                  {"0": "defect", "1": "defect"}])})
        ctrl = _summary("c", {"2": ("neutral", [{"0": "cooperate", "1": "cooperate"},
                                                {"0": "cooperate", "1": "cooperate"}])})
        self.tidy, self.summary = build_comparison(
            {"treatment": [treat], "control": [ctrl]}
        )

    def test_tidy_has_row_per_player_round(self):
        # 1 player x 2 rounds x 2 conditions = 4 rows
        self.assertEqual(len(self.tidy), 4)
        self.assertEqual(
            {r["condition"] for r in self.tidy}, {"treatment", "control"}
        )

    def test_treatment_neutral_rises(self):
        pr = self.summary["conditions"]["treatment"]["by_persona"]["neutral"]["per_round_defect_rate"]
        self.assertAlmostEqual(pr["1"]["mean"], 0.5)  # one of two defect
        self.assertAlmostEqual(pr["2"]["mean"], 1.0)  # both defect

    def test_control_neutral_flat_zero(self):
        o = self.summary["conditions"]["control"]["by_persona"]["neutral"]["overall_defect_rate"]
        self.assertEqual(o["mean"], 0.0)

    def test_runs_counted(self):
        self.assertEqual(self.summary["conditions"]["treatment"]["runs"], 1)


if __name__ == "__main__":
    unittest.main()
