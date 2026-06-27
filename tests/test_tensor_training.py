import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from claq.constraints import ConstraintConfig, EmpiricalConditionalMIScorer
from claq.training import (
    ClaqObjectiveConfig,
    HistorySamplingConfig,
    build_claq_models,
    run_claq_tensor_epoch,
)


class TensorTrainingTest(unittest.TestCase):
    def test_precomputed_response_training(self):
        torch.manual_seed(4)
        answers = torch.where(torch.randn(24, 5) > 0, 1.0, -1.0)
        labels = (answers[:, 0] > 0).long()
        sensitive = (answers[:, 1] > 0).long()
        loader = DataLoader(TensorDataset(answers, labels, sensitive), batch_size=8)
        scorer = EmpiricalConditionalMIScorer(answers, labels, sensitive, min_support=1)
        actor, classifier, sensitive_head = build_claq_models(
            5,
            2,
            torch.device("cpu"),
            actor_eps=0.7,
            condition_sensitive_on_y=True,
            hidden_dims=(16, 8),
        )
        optimizer = torch.optim.Adam(
            list(actor.parameters()) + list(classifier.parameters()) + list(sensitive_head.parameters()),
            lr=1e-3,
        )
        objective = ClaqObjectiveConfig(
            lambda_sensitive=0.2,
            lambda_cost=0.1,
            lambda_fairness=0.1,
            condition_sensitive_on_y=True,
            constraints=ConstraintConfig(query_costs=torch.ones(5), cost_budget=2.0),
            mi_scorer=scorer,
            rollout_steps=2,
            start_from_sampled_history=False,
        )
        metrics = run_claq_tensor_epoch(
            loader=loader,
            actor=actor,
            classifier=classifier,
            s_head=sensitive_head,
            optimizer=optimizer,
            objective_config=objective,
            history_config=HistorySamplingConfig(0, 0, False),
            device=torch.device("cpu"),
            train=True,
        )
        self.assertLessEqual(metrics["mean_queries"], 2.0)
        self.assertIn("eo_violation", metrics)


if __name__ == "__main__":
    unittest.main()
