from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from claq.core.checkpoints import load_claq_bundle_checkpoint
from claq.models import Network


class CheckpointTests(unittest.TestCase):
    def test_load_infers_hidden_dimensions(self):
        actor = Network(query_size=4, output_size=4, eps=1.0, hidden_dims=(12, 7))
        classifier = Network(query_size=4, output_size=3, eps=None, hidden_dims=(11, 6))
        sensitive = Network(query_size=7, output_size=1, eps=None, hidden_dims=(10, 5))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.pt"
            torch.save({
                "actor_state_dict": actor.state_dict(),
                "classifier_state_dict": classifier.state_dict(),
                "s_head_state_dict": sensitive.state_dict(),
            }, path)
            bundle = load_claq_bundle_checkpoint(path, device=torch.device("cpu"))
        self.assertEqual(bundle["actor"].query_size, 4)
        self.assertEqual(bundle["classifier"].output_dim, 3)
        self.assertEqual(bundle["sensitive_head"].query_size, 7)

    def test_load_restores_selected_actor_temperature(self):
        actor = Network(query_size=3, output_size=3, eps=0.25, hidden_dims=(8, 4))
        classifier = Network(query_size=3, output_size=2, eps=None, hidden_dims=(8, 4))
        sensitive = Network(query_size=5, output_size=1, eps=None, hidden_dims=(8, 4))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.pt"
            torch.save({
                "actor_state_dict": actor.state_dict(),
                "classifier_state_dict": classifier.state_dict(),
                "s_head_state_dict": sensitive.state_dict(),
                "model_config": {"actor_temperature_selected": 0.25},
            }, path)
            bundle = load_claq_bundle_checkpoint(path, device=torch.device("cpu"))
        self.assertAlmostEqual(bundle["actor"].eps, 0.25)


if __name__ == "__main__":
    unittest.main()
