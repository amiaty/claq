from pathlib import Path
from tempfile import TemporaryDirectory
import importlib.util
import unittest

import torch

from claq.models import Network


def _load_core_module(name: str):
    path = Path(__file__).parents[1] / "claq" / "core" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_claq_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


answer_cache = _load_core_module("answer_cache")
checkpoints = _load_core_module("checkpoints")
load_answer_cache = answer_cache.load_answer_cache
make_cached_answer_loader = answer_cache.make_cached_answer_loader
save_answer_cache = answer_cache.save_answer_cache
load_run_bundle = checkpoints.load_run_bundle
save_bundle_checkpoint = checkpoints.save_bundle_checkpoint


class ArtifactTests(unittest.TestCase):
    def test_compact_policy_checkpoint_round_trip(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "run.pt"
            actor = Network(query_size=5, output_size=5, eps=0.2)
            classifier = Network(query_size=5, output_size=3, eps=None)
            s_head = Network(query_size=8, output_size=1, eps=None)

            save_bundle_checkpoint(
                path,
                actor=actor,
                classifier=classifier,
                s_head=s_head,
                metadata={"run_name": "test", "seed": 2},
            )
            bundle = load_run_bundle(
                path,
                device=torch.device("cpu"),
                max_queries=5,
                num_classes=3,
                actor_eps=0.2,
            )

            self.assertEqual(bundle["meta"]["format_version"], 1)
            self.assertEqual(bundle["s_head"].query_size, 8)
            self.assertTrue(
                torch.equal(
                    actor.state_dict()["layer1.weight"],
                    bundle["actor"].state_dict()["layer1.weight"],
                )
            )

    def test_legacy_sensitive_head_alias_loads(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            actor = Network(query_size=4, output_size=4, eps=1.0)
            classifier = Network(query_size=4, output_size=2, eps=None)
            s_head = Network(query_size=6, output_size=1, eps=None)
            torch.save(
                {
                    "actor_state_dict": actor.state_dict(),
                    "classifier_state_dict": classifier.state_dict(),
                    "sensitive_head_state_dict": s_head.state_dict(),
                },
                path,
            )
            bundle = load_run_bundle(
                path,
                device=torch.device("cpu"),
                max_queries=4,
                num_classes=2,
            )
            self.assertIsNotNone(bundle["s_head"])
            self.assertEqual(bundle["s_head"].query_size, 6)

    def test_answer_cache_is_compact_and_validated(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "answers.pt"
            answers = torch.tensor([[-1, 1, -1], [1, 1, -1]], dtype=torch.float32)
            labels = torch.tensor([0, 1])
            sensitive = torch.tensor([0.25, 0.75])
            save_answer_cache(
                path,
                answers=answers,
                labels=labels,
                sensitive_targets=sensitive,
                metadata={"checkpoint": "qa.pt"},
            )
            payload = load_answer_cache(
                path, expected_metadata={"checkpoint": "qa.pt"}
            )
            self.assertEqual(payload["answers"].dtype, torch.int8)
            loader = make_cached_answer_loader(
                payload, batch_size=2, shuffle=False
            )
            batch = next(iter(loader))
            self.assertTrue(
                torch.equal(batch["precomputed_answers"].float(), answers)
            )
            with self.assertRaises(ValueError):
                load_answer_cache(
                    path, expected_metadata={"checkpoint": "different.pt"}
                )


if __name__ == "__main__":
    unittest.main()
