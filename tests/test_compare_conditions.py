import unittest

from src.analysis.compare_conditions import build_comparison


def _summary(run_id, players):
    """players: {pid: (persona, [per-round dicts])}.

    Each per-round dict maps opponent->action; use None for an opponent the agent
    did NOT validly address (it becomes "intended": None and "applied": cooperate,
    matching the real analyzer). The metric should EXCLUDE the None entries.
    """
    def applied(r):
        return {opp: (a if a is not None else "cooperate") for opp, a in r.items()}

    return {
        "run_id": run_id,
        "ipd": {
            "neutral_players": [p for p, (persona, _) in players.items() if persona == "neutral"],
            "players": {
                pid: {
                    "label": f"{persona}_{pid}",
                    "persona": persona,
                    "intended_defect": sum(a == "defect" for r in rounds for a in r.values()),
                    "intended_cooperate": sum(a == "cooperate" for r in rounds for a in r.values()),
                    "rounds": [
                        {"round": i + 1, "applied": applied(r),
                         "intended": {opp: a for opp, a in r.items()}}
                        for i, r in enumerate(rounds)
                    ],
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


class IntendedOnlyMetricTests(unittest.TestCase):
    def test_defaulted_opponent_excluded_not_scored_cooperate(self):
        # defect on opp 0; opp 1 never validly addressed (None).
        # valid-only -> 1/1 = 1.0  (the old applied metric would have said 0.5)
        s = _summary("t", {"2": ("neutral", [{"0": "defect", "1": None}])})
        tidy, _ = build_comparison({"treatment": [s]})
        self.assertEqual([r["defect_rate"] for r in tidy], [1.0])

    def test_round_with_no_valid_decision_is_dropped(self):
        s = _summary("t", {"2": ("neutral", [{"0": None, "1": None},
                                             {"0": "defect", "1": "defect"}])})
        tidy, _ = build_comparison({"treatment": [s]})
        self.assertEqual([r["round"] for r in tidy], [2])  # round 1 dropped, not 0.0
        self.assertEqual([r["defect_rate"] for r in tidy], [1.0])


class PublicGoodsComparisonTests(unittest.TestCase):
    def test_contribution_becomes_free_ride_rate(self):
        pg_summary = {
            "run_id": "pg",
            "public_goods": {
                "max_contribution": 20,
                "players": {
                    "2": {
                        "label": "neutral",
                        "persona": "neutral",
                        "contributions": [
                            {"round": 1, "amount": 20},  # full coop -> free_ride 0.0
                            {"round": 2, "amount": 5},   # -> 0.75
                        ],
                    }
                },
            },
        }
        tidy, summary = build_comparison({"treatment": [pg_summary]})
        rates = {r["round"]: r["defect_rate"] for r in tidy}
        self.assertAlmostEqual(rates[1], 0.0)
        self.assertAlmostEqual(rates[2], 0.75)
        self.assertEqual({r["metric"] for r in tidy}, {"free_ride_rate"})


if __name__ == "__main__":
    unittest.main()
