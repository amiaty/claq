import unittest

import torch

from claq.constraints import (
    ConstraintConfig,
    EmpiricalConditionalMIScorer,
    build_static_proxy_admissible_mask,
    build_unavailable_action_mask,
)


class ConstraintTests(unittest.TestCase):
    def setUp(self):
        # q0 determines Y, q1 determines S within each Y, q2 is constant.
        self.answers = torch.tensor(
            [
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        self.labels = torch.tensor([0, 0, 1, 1])
        self.sensitive = torch.tensor([0, 1, 0, 1])
        self.scorer = EmpiricalConditionalMIScorer(
            self.answers,
            self.labels,
            self.sensitive,
            alpha=0.0,
            min_support=1,
        )

    def test_information_scores(self):
        scores = self.scorer.global_scores
        self.assertAlmostEqual(float(scores.label_information[0]), 1.0, places=6)
        self.assertAlmostEqual(float(scores.sensitive_leakage[1]), 1.0, places=6)
        self.assertAlmostEqual(float(scores.label_information[2]), 0.0, places=6)

    def test_static_proxy_rule(self):
        admissible = build_static_proxy_admissible_mask(
            self.scorer,
            leakage_threshold=0.5,
            label_information_threshold=0.5,
        )
        self.assertTrue(bool(admissible[0]))
        self.assertFalse(bool(admissible[1]))
        self.assertTrue(bool(admissible[2]))

    def test_cost_and_leakage_feasibility(self):
        queried = torch.zeros((1, 3))
        history = torch.zeros((1, 3))
        scores = self.scorer.score_batch(queried, history)
        config = ConstraintConfig(
            query_costs=torch.tensor([1.0, 2.0, 3.0]),
            cost_budget=2.0,
            leakage_budget=0.5,
        )
        unavailable = build_unavailable_action_mask(
            queried,
            config=config,
            scores=scores,
            cumulative_cost=torch.zeros(1),
            cumulative_leakage=torch.zeros(1),
        )
        self.assertFalse(bool(unavailable[0, 0]))
        self.assertTrue(bool(unavailable[0, 1]))
        self.assertTrue(bool(unavailable[0, 2]))


if __name__ == "__main__":
    unittest.main()
