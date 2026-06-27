import unittest

import torch
import torch.nn.functional as F

from claq.constraints import ConstraintConfig, EmpiricalConditionalMIScorer
from claq.core.runtime import rollout_claq_batch


class FirstAvailableActor(torch.nn.Module):
    def forward(self, state, mask, unavailable_mask=None):
        blocked = mask > 0.5
        if unavailable_mask is not None:
            blocked = blocked | unavailable_mask
        available = ~blocked
        has = available.any(dim=1)
        index = available.to(torch.int64).argmax(dim=1)
        result = F.one_hot(index, num_classes=state.size(1)).to(state.dtype)
        return result * has.to(state.dtype).view(-1, 1)


class ConstantClassifier(torch.nn.Module):
    def forward(self, state):
        return torch.stack([state.sum(dim=1), -state.sum(dim=1)], dim=1)


class HighConfidenceClassifier(torch.nn.Module):
    def forward(self, state):
        return torch.stack([torch.full((state.size(0),), 10.0), torch.zeros(state.size(0))], dim=1)


class RuntimeTests(unittest.TestCase):
    def test_cost_budget_is_enforced(self):
        answers = torch.tensor([[1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]])
        config = ConstraintConfig(
            query_costs=torch.tensor([1.0, 2.0, 3.0]),
            cost_budget=1.0,
        )
        result = rollout_claq_batch(
            answers=answers,
            actor=FirstAvailableActor(),
            classifier=ConstantClassifier(),
            max_steps=3,
            constraint_config=config,
        )
        self.assertTrue(torch.equal(result["query_counts"], torch.tensor([1, 1])))
        self.assertTrue(torch.allclose(result["cumulative_cost"], torch.ones(2)))

    def test_dynamic_proxy_screen_is_enforced(self):
        reference_answers = torch.tensor(
            [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
        )
        labels = torch.tensor([0, 0, 1, 1])
        sensitive = torch.tensor([0, 1, 0, 1])
        scorer = EmpiricalConditionalMIScorer(
            reference_answers, labels, sensitive, alpha=0.0, min_support=1
        )
        config = ConstraintConfig(
            dynamic_proxy_screen=True,
            proxy_leakage_threshold=0.5,
            proxy_label_threshold=0.5,
        )
        result = rollout_claq_batch(
            answers=reference_answers,
            actor=FirstAvailableActor(),
            classifier=ConstantClassifier(),
            max_steps=2,
            constraint_config=config,
            mi_scorer=scorer,
        )
        # q1 is the pure sensitive proxy and must never be executed.
        self.assertFalse(bool((result["actions"] == 1).any().item()))

    def test_confidence_stopping_respects_minimum_steps(self):
        answers = torch.tensor([[1.0, -1.0]])
        immediate = rollout_claq_batch(
            answers=answers,
            actor=FirstAvailableActor(),
            classifier=HighConfidenceClassifier(),
            max_steps=2,
            confidence_threshold=0.9,
            minimum_steps=0,
        )
        delayed = rollout_claq_batch(
            answers=answers,
            actor=FirstAvailableActor(),
            classifier=HighConfidenceClassifier(),
            max_steps=2,
            confidence_threshold=0.9,
            minimum_steps=1,
        )
        self.assertEqual(int(immediate["query_counts"][0]), 0)
        self.assertEqual(int(delayed["query_counts"][0]), 1)
        self.assertEqual(immediate["stop_reason"][0], "confidence")
        self.assertEqual(delayed["stop_reason"][0], "confidence")


if __name__ == "__main__":
    unittest.main()
