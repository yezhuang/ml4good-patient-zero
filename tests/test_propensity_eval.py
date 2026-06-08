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


if __name__ == "__main__":
    unittest.main()
