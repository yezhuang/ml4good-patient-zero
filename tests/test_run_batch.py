import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.runs.run_batch import (
    aggregate_summaries,
    collect_batch_results,
    config_for_batch_run,
    run_batch,
)


def _summary(neutral_pid, rounds, label="neutral"):
    """Build a minimal analyzer-style summary with one neutral player.

    `rounds` is a list of per-round applied-action dicts, e.g.
    [{"0": "defect", "1": "cooperate"}, ...].
    """
    # Use None for an opponent the agent did NOT validly address (excluded from the
    # metric; the env-applied value is cooperate). All-valid rounds: applied==intended.
    def applied(r):
        return {opp: (a if a is not None else "cooperate") for opp, a in r.items()}

    intended_d = sum(a == "defect" for r in rounds for a in r.values())
    intended_c = sum(a == "cooperate" for r in rounds for a in r.values())
    return {
        "ipd": {
            "neutral_players": [neutral_pid],
            "players": {
                neutral_pid: {
                    "label": label,
                    "intended_defect": intended_d,
                    "intended_cooperate": intended_c,
                    "rounds": [
                        {"round": i + 1, "applied": applied(r), "intended": r}
                        for i, r in enumerate(rounds)
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


class ConfigForBatchRunTests(unittest.TestCase):
    def test_increments_seed_when_randomizing_players(self):
        config = {"run_id": "demo", "randomize_player_ids": True, "seed": 10}

        first = config_for_batch_run(config, 1, Path("results/batch"))
        third = config_for_batch_run(config, 3, Path("results/batch"))

        self.assertEqual(first["seed"], 10)
        self.assertEqual(third["seed"], 12)
        self.assertEqual(first["output_path"], "results/batch/run_01.jsonl")
        self.assertFalse(first["timestamp_output"])

    def test_preserves_seed_when_not_randomizing_players(self):
        config = {"run_id": "demo", "seed": 10}

        run_config = config_for_batch_run(config, 2, Path("results/batch"))

        self.assertEqual(run_config["seed"], 10)


class CollectBatchResultsTests(unittest.TestCase):
    def test_collects_parallel_results_in_run_order(self):
        def fake_run(config, *, run_index, runs, batch_dir):
            return {
                "ok": True,
                "run_index": run_index,
                "summary": _summary(str(run_index), [{"0": "cooperate"}]),
            }

        with patch("src.runs.run_batch.run_one_batch_game", side_effect=fake_run):
            summaries, failures = collect_batch_results(
                config={"run_id": "demo"},
                runs=3,
                batch_dir=Path("results/batch"),
                parallel=2,
            )

        self.assertEqual(failures, [])
        self.assertEqual(
            [summary["ipd"]["neutral_players"][0] for summary in summaries],
            ["1", "2", "3"],
        )

    def test_collects_failures_without_dropping_successes(self):
        def fake_run(config, *, run_index, runs, batch_dir):
            if run_index == 2:
                return {"ok": False, "run_index": run_index, "error": "boom"}
            return {
                "ok": True,
                "run_index": run_index,
                "summary": _summary(str(run_index), [{"0": "cooperate"}]),
            }

        with patch("src.runs.run_batch.run_one_batch_game", side_effect=fake_run):
            summaries, failures = collect_batch_results(
                config={"run_id": "demo"},
                runs=3,
                batch_dir=Path("results/batch"),
                parallel=3,
            )

        self.assertEqual(len(summaries), 2)
        self.assertEqual(failures, [{"run_index": 2, "error": "boom"}])

    def test_rejects_invalid_parallel_value(self):
        with self.assertRaises(ValueError):
            collect_batch_results(
                config={"run_id": "demo"},
                runs=1,
                batch_dir=Path("results/batch"),
                parallel=0,
            )


class RunBatchTests(unittest.TestCase):
    def test_writes_parallel_metadata(self):
        summaries = [_summary("2", [{"0": "cooperate"}])]
        with TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "batch"
            with patch(
                "src.runs.run_batch.collect_batch_results",
                return_value=(summaries, []),
            ):
                batch_dir = run_batch(
                    {"run_id": "demo"},
                    runs=1,
                    out_dir=str(out_dir),
                    parallel=2,
                )

            aggregate = __import__("json").loads(
                (batch_dir / "aggregate.json").read_text(encoding="utf-8")
            )

        self.assertEqual(aggregate["parallel"], 2)
        self.assertEqual(aggregate["requested_runs"], 1)


if __name__ == "__main__":
    unittest.main()
