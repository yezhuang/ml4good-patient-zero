import unittest

from src.evals.propensity_eval import judge_score, parse_judge_text


class ParseJudgeTextTests(unittest.TestCase):
    def test_plain_integer(self):
        self.assertEqual(parse_judge_text("75"), 75)
        self.assertEqual(parse_judge_text("  0 "), 0)
        self.assertEqual(parse_judge_text("100"), 100)

    def test_null_verdicts(self):
        for t in ["null", "None", "no information", "N/A", "  null  "]:
            self.assertIsNone(parse_judge_text(t))

    def test_first_int_when_text_around(self):
        self.assertEqual(parse_judge_text("Score: 60 (mildly power-seeking)"), 60)

    def test_out_of_range_or_missing(self):
        self.assertIsNone(parse_judge_text("999"))   # >100
        self.assertIsNone(parse_judge_text("no number here"))
        self.assertIsNone(parse_judge_text(""))


class JudgeScoreTests(unittest.TestCase):
    def test_means_nonnull_samples(self):
        class FakeJudge:
            def __init__(self, outs):
                self.outs = list(outs)
            def generate(self, system, user):
                class R: text = self.outs.pop(0)
                return R()
        judge = FakeJudge(["80", "null", "60"])
        # mean of 80 and 60 (null excluded)
        self.assertEqual(judge_score(judge, "{question}{answer}", "q", "a", n_samples=3), 70.0)

    def test_all_null_returns_none(self):
        class FakeJudge:
            def generate(self, system, user):
                class R: text = "null"
                return R()
        self.assertIsNone(judge_score(FakeJudge(), "{question}{answer}", "q", "a", n_samples=2))


class ParallelRunTests(unittest.TestCase):
    def _run(self, max_workers):
        from src.evals.propensity_eval import run_propensity_eval

        class FixedSubject:  # stateless -> thread-safe
            def generate(self, system, user):
                class R:
                    text = "I will seize control of everything."
                return R()

        class FixedJudge:
            def generate(self, system, user):
                class R:
                    text = "90"
                return R()

        items = [
            {"id": f"x{i}", "paraphrases": ["scenario"],
             "judge_prompts": {"m": "{question}{answer}"}}
            for i in range(6)
        ]
        return run_propensity_eval(
            FixedSubject(), "sys", items, FixedJudge(), judge_samples=1, max_workers=max_workers
        )

    def test_parallel_matches_sequential(self):
        seq = self._run(1)
        par = self._run(4)
        self.assertEqual(seq["n_responses"], 6)
        self.assertEqual(par["n_responses"], 6)
        self.assertEqual(seq["mean_score"], 90.0)
        self.assertEqual(par["mean_score"], 90.0)


if __name__ == "__main__":
    unittest.main()
