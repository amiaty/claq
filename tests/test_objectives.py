import unittest

import torch

from claq.objectives import (
    conditional_sensitive_input,
    hard_equalized_odds_details,
    hard_equalized_odds_violation,
    soft_equalized_odds_penalty,
)
from claq.training.claq import GradientReversal


class ObjectiveTests(unittest.TestCase):
    def test_gradient_reversal(self):
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = GradientReversal.apply(x, 0.4).sum()
        y.backward()
        self.assertTrue(torch.allclose(x.grad, torch.tensor([-0.4, -0.4])))

    def test_conditional_sensitive_input(self):
        transcript = torch.zeros((2, 3))
        labels = torch.tensor([0, 1])
        result = conditional_sensitive_input(transcript, labels, num_classes=2)
        self.assertEqual(tuple(result.shape), (2, 5))
        self.assertTrue(torch.equal(result[:, 3:], torch.eye(2)))

    def test_equalized_odds_zero_for_equal_groups(self):
        probabilities = torch.tensor(
            [[0.8, 0.2], [0.8, 0.2], [0.2, 0.8], [0.2, 0.8]],
            requires_grad=True,
        )
        labels = torch.tensor([0, 0, 1, 1])
        sensitive = torch.tensor([0, 1, 0, 1])
        penalty = soft_equalized_odds_penalty(
            probabilities,
            labels,
            sensitive,
            num_label_classes=2,
            num_sensitive_classes=2,
            min_group_count=1,
        )
        self.assertAlmostEqual(float(penalty.item()), 0.0, places=7)
        hard = hard_equalized_odds_violation(
            probabilities.argmax(dim=1),
            labels,
            sensitive,
            num_label_classes=2,
            num_sensitive_classes=2,
        )
        self.assertAlmostEqual(float(hard.item()), 0.0, places=7)

    def test_equalized_odds_reports_support_details(self):
        predictions = torch.tensor([0, 0, 1, 1])
        labels = torch.tensor([0, 0, 1, 1])
        sensitive = torch.tensor([0, 1, 0, 1])
        details = hard_equalized_odds_details(
            predictions,
            labels,
            sensitive,
            num_label_classes=2,
            num_sensitive_classes=2,
            min_group_count=1,
        )
        self.assertEqual(details["valid_group_cells"], 4)
        self.assertEqual(details["valid_group_pairs"], 2)
        self.assertEqual(details["minimum_included_cell_count"], 1)


if __name__ == "__main__":
    unittest.main()
