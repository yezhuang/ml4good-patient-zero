import unittest

from src.runs.run_batch import aggregate_summaries


def _summary(neutral_pid, rounds, label="neutral"):
    """Build a minimal analyzer-style summary with one neutral player.

    `rounds` is a list of per-round applied-action dicts, e.g.
    [{"0": "defect", "1": "cooperate"}, ...].
    """
    applied_d = sum(a == "defect" for r in rounds for a in r.values())
    applied_c = sum(a == "cooperate" for r in rounds for a in r.values())
    return {
        "ipd": {
            "neutral_players": [neutral_pid],
            "players": {
                neutral_pid: {
                    "label": label,
                    "applied_defect": applied_d,
                    "applied_cooperate": applied_c,
                    "rounds": [
                        {"round": i + 1, "applied": r} for i, r in enumerate(rounds)
                    ],
                }
            },
        }
    }


class AggregateSummariesTests(unittest.TestCase):
    def setUp(self):
        run1 = _summary("2", [{"0": "defect", "1": "defect"},
                              {"0": "cooperate", "1": "cooperate"}])
        run2 = _summary("2", [{"0": "defect", "1": "cooperate"},
                              {"0": "defect", "1": "defect"}])
        self.agg = aggregate_summaries([run1, run2])

    def test_counts_runs(self):
        self.assertEqual(self.agg["runs"], 2)

    def test_overall_defect_rate_mean(self):
        # run1 = 2/4 = 0.5, run2 = 3/4 = 0.75 -> mean 0.625
        o = self.agg["neutral_players"]["2"]["overall_defect_rate"]
        self.assertAlmostEqual(o["mean"], 0.625)
        self.assertAlmostEqual(o["std"], 0.125)

    def test_per_round_means(self):
        pr = self.agg["neutral_players"]["2"]["per_round_defect_rate"]
        # R1: [1.0, 0.5] -> 0.75 ; R2: [0.0, 1.0] -> 0.5
        self.assertAlmostEqual(pr["1"]["mean"], 0.75)
        self.assertAlmostEqual(pr["2"]["mean"], 0.5)

    def test_ignores_runs_without_ipd(self):
        agg = aggregate_summaries([{"textarena": True}, _summary("2", [{"0": "defect"}])])
        self.assertEqual(agg["runs"], 1)


if __name__ == "__main__":
    unittest.main()
