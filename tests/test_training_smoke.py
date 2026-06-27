import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from claq.constraints import ConstraintConfig, EmpiricalConditionalMIScorer
from claq.core.runtime import concept_answers_from_image_features
from claq.training import (
    ClaqObjectiveConfig,
    HistorySamplingConfig,
    build_claq_models,
    run_claq_epoch,
)


class DummyClip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def encode_image(self, images):
        return images


class DummyAnswerer(torch.nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.linear = torch.nn.Linear(2 * feature_dim, 1)

    def forward(self, inputs):
        return self.linear(inputs)


class TrainingSmokeTest(unittest.TestCase):
    def test_complete_objective_runs(self):
        torch.manual_seed(3)
        n, feature_dim, num_queries = 16, 4, 4
        images = torch.randn(n, feature_dim)
        labels = (images[:, 0] > 0).long()
        concept_targets = (torch.randn(n, num_queries) > 0).float()
        loader = DataLoader(TensorDataset(images, labels, concept_targets), batch_size=8)
        model_clip = DummyClip()
        dictionary = torch.randn(feature_dim, num_queries)
        dictionary = dictionary / dictionary.norm(dim=0, keepdim=True)
        answerer = DummyAnswerer(feature_dim)
        device = torch.device("cpu")
        actor, classifier, s_head = build_claq_models(
            num_queries,
            2,
            device,
            actor_eps=0.7,
            condition_sensitive_on_y=True,
            hidden_dims=(16, 8),
        )
        optimizer = torch.optim.Adam(
            list(actor.parameters()) + list(classifier.parameters()) + list(s_head.parameters()),
            lr=1e-3,
        )
        with torch.no_grad():
            normalized = images / images.norm(dim=1, keepdim=True)
            full_answers = concept_answers_from_image_features(
                normalized, dictionary, answerer, device
            )
        sensitive = concept_targets[:, 0].long()
        scorer = EmpiricalConditionalMIScorer(
            full_answers, labels, sensitive, min_support=1
        )
        objective = ClaqObjectiveConfig(
            lambda_sensitive=0.2,
            lambda_cost=0.1,
            lambda_fairness=0.1,
            condition_sensitive_on_y=True,
            constraints=ConstraintConfig(
                query_costs=torch.ones(num_queries),
                cost_budget=2.0,
                leakage_budget=2.0,
            ),
            mi_scorer=scorer,
            rollout_steps=2,
            start_from_sampled_history=False,
        )
        metrics = run_claq_epoch(
            loader=loader,
            actor=actor,
            classifier=classifier,
            s_head=s_head,
            optimizer=optimizer,
            model_clip=model_clip,
            dictionary=dictionary,
            answering_model=answerer,
            sens_idx=torch.tensor([0]),
            history_config=HistorySamplingConfig(0, 0, False),
            clip_device=device,
            train_device=device,
            threshold_for_binarization=0.0,
            train=True,
            sensitive_target_mode="hard",
            objective_config=objective,
        )
        self.assertIn("macro_f1", metrics)
        self.assertIn("eo_violation", metrics)
        self.assertLessEqual(metrics["mean_queries"], 2.0)


if __name__ == "__main__":
    unittest.main()
