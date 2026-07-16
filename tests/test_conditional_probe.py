import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn as nn


def _load_analysis_module(name: str):
    path = Path(__file__).parents[1] / "claq" / "analysis" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_claq_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


conditional_probe = _load_analysis_module("conditional_probe")
joint_eval = _load_analysis_module("joint_eval")
conditional_entropy_bits = conditional_probe.conditional_entropy_bits
fit_conditional_probe = conditional_probe.fit_conditional_probe
fixed_horizon_rollout = joint_eval.fixed_horizon_rollout
summarize_fixed_horizon = joint_eval.summarize_fixed_horizon


class _FirstAvailableActor(nn.Module):
    def forward(self, knowledge_state, mask):
        index = (mask < 0.5).float().argmax(dim=1)
        return torch.nn.functional.one_hot(
            index, num_classes=mask.shape[1]
        ).float()


class _SignClassifier(nn.Module):
    def forward(self, knowledge_state):
        return torch.stack((-knowledge_state[:, 0], knowledge_state[:, 0]), dim=1)


class ConditionalProbeTests(unittest.TestCase):
    def test_conditional_entropy(self):
        sensitive = np.array([0, 1, 0, 1, 0, 0])
        labels = np.array([0, 0, 1, 1, 2, 2])
        self.assertAlmostEqual(conditional_entropy_bits(sensitive, labels), 2 / 3)

    def test_probe_detects_recoverable_sensitive_attribute(self):
        rng = np.random.default_rng(7)
        sample_count = 300
        labels = rng.integers(0, 3, sample_count)
        sensitive = rng.integers(0, 2, sample_count)
        states = np.zeros((sample_count, 4))
        states[:, 0] = 2 * sensitive - 1
        states[:, 1:] = rng.normal(scale=0.1, size=(sample_count, 3))

        result = fit_conditional_probe(
            train_states=states[:180],
            train_labels=labels[:180],
            train_sensitive=sensitive[:180],
            validation_states=states[180:240],
            validation_labels=labels[180:240],
            validation_sensitive=sensitive[180:240],
            test_states=states[240:],
            test_labels=labels[240:],
            test_sensitive=sensitive[240:],
            num_classes=3,
        )

        self.assertGreater(result["probe_leakage_bits"], 0.8)
        self.assertGreater(result["test_probe_accuracy"], 0.95)
        self.assertIn(result["selected_c"], (0.01, 0.1, 1.0, 10.0))

    def test_probe_rejects_nonbinary_sensitive_targets(self):
        states = np.zeros((4, 2))
        with self.assertRaises(ValueError):
            fit_conditional_probe(
                train_states=states,
                train_labels=[0, 0, 1, 1],
                train_sensitive=[0, 1, 2, 0],
                validation_states=states,
                validation_labels=[0, 0, 1, 1],
                validation_sensitive=[0, 1, 0, 1],
                test_states=states,
                test_labels=[0, 0, 1, 1],
                test_sensitive=[0, 1, 0, 1],
                num_classes=2,
            )

    def test_fixed_horizon_rollout_counts_cost_and_sensitive_queries(self):
        rollout = fixed_horizon_rollout(
            actor=_FirstAvailableActor(),
            classifier=_SignClassifier(),
            answers=torch.tensor([[1, -1, 1], [-1, 1, -1]]),
            labels=torch.tensor([1, 0]),
            sensitive_targets=torch.tensor([1, 0]),
            cost_vector=torch.tensor([1.5, 2.0, 1.0]),
            sensitive_mask=torch.tensor([1.0, 0.0, 0.0]),
            horizon=2,
            device=torch.device("cpu"),
        )
        summary = summarize_fixed_horizon(
            rollout, horizon=2, include_macro_f1=True
        )

        self.assertEqual(summary["accuracy"], 1.0)
        self.assertEqual(summary["macro_f1"], 1.0)
        self.assertAlmostEqual(summary["mean_cumulative_cost"], 3.5)
        self.assertAlmostEqual(summary["sensitive_query_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
