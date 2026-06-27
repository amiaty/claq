from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from claq.precomputed import (
    PrecomputedDataset,
    PrecomputedSplit,
    load_precomputed_dataset,
    normalize_binary_responses,
    save_precomputed_dataset,
)


class PrecomputedTests(unittest.TestCase):
    def _dataset(self) -> PrecomputedDataset:
        responses = torch.tensor([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=torch.float32)
        labels = torch.tensor([0, 1, 1, 0])
        sensitive = torch.tensor([0, 1, 0, 1])
        split = PrecomputedSplit(normalize_binary_responses(responses), labels, sensitive)
        return PrecomputedDataset(
            train=split,
            validation=split,
            test=split,
            query_names=("a", "b"),
            query_set_mask=torch.tensor([True, False]),
            query_costs=torch.ones(2),
            admissible_mask=torch.ones(2, dtype=torch.bool),
            justification_mask=torch.zeros(2, dtype=torch.bool),
            metadata={"dataset": "toy"},
        )

    def test_round_trip(self):
        dataset = self._dataset()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.npz"
            save_precomputed_dataset(path, dataset)
            loaded = load_precomputed_dataset(path)
        self.assertEqual(loaded.query_names, dataset.query_names)
        self.assertTrue(torch.equal(loaded.train.responses, dataset.train.responses))
        self.assertTrue(torch.equal(loaded.query_set_mask, dataset.query_set_mask))
        self.assertEqual(loaded.metadata["dataset"], "toy")

    def test_zero_is_reserved_for_unqueried(self):
        converted = normalize_binary_responses(torch.tensor([[0.0, 1.0]]))
        self.assertTrue(torch.equal(converted, torch.tensor([[-1.0, 1.0]])))

    def test_class_counts_are_inferred_from_training_only(self):
        dataset = self._dataset()
        validation = PrecomputedSplit(
            dataset.validation.responses[:2],
            torch.tensor([0, 0]),
            torch.tensor([0, 0]),
        )
        modified = PrecomputedDataset(
            train=dataset.train,
            validation=validation,
            test=validation,
            query_names=dataset.query_names,
            query_set_mask=dataset.query_set_mask,
            query_costs=dataset.query_costs,
            admissible_mask=dataset.admissible_mask,
            justification_mask=dataset.justification_mask,
            metadata=dataset.metadata,
        )
        modified.validate()
        self.assertEqual(modified.num_label_classes, 2)
        self.assertEqual(modified.num_sensitive_classes, 2)

    def test_duplicate_query_names_are_rejected(self):
        dataset = self._dataset()
        duplicate = PrecomputedDataset(
            train=dataset.train,
            validation=dataset.validation,
            test=dataset.test,
            query_names=("a", "a"),
            query_set_mask=dataset.query_set_mask,
            query_costs=dataset.query_costs,
            admissible_mask=dataset.admissible_mask,
            justification_mask=dataset.justification_mask,
            metadata=dataset.metadata,
        )
        with self.assertRaises(ValueError):
            duplicate.validate()

    def test_invalid_response_values_are_rejected(self):
        dataset = self._dataset()
        invalid_split = PrecomputedSplit(
            torch.tensor([[0.2, 1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, -1.0]]),
            dataset.train.labels,
            dataset.train.sensitive,
        )
        invalid = PrecomputedDataset(
            train=invalid_split,
            validation=dataset.validation,
            test=dataset.test,
            query_names=dataset.query_names,
            query_set_mask=dataset.query_set_mask,
            query_costs=dataset.query_costs,
            admissible_mask=dataset.admissible_mask,
            justification_mask=dataset.justification_mask,
            metadata=dataset.metadata,
        )
        with self.assertRaises(ValueError):
            invalid.validate()

    def test_legacy_object_archive_requires_explicit_opt_in(self):
        dataset = self._dataset()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.npz"
            np.savez_compressed(
                path,
                responses_train=dataset.train.responses.numpy(),
                y_train=dataset.train.labels.numpy(),
                s_train=dataset.train.sensitive.numpy(),
                responses_validation=dataset.validation.responses.numpy(),
                y_validation=dataset.validation.labels.numpy(),
                s_validation=dataset.validation.sensitive.numpy(),
                responses_test=dataset.test.responses.numpy(),
                y_test=dataset.test.labels.numpy(),
                s_test=dataset.test.sensitive.numpy(),
                query_names=np.asarray(dataset.query_names, dtype=object),
                query_set_mask=dataset.query_set_mask.numpy(),
                query_costs=dataset.query_costs.numpy(),
                admissible_mask=dataset.admissible_mask.numpy(),
                justification_mask=dataset.justification_mask.numpy(),
                metadata=np.asarray(dataset.metadata, dtype=object),
            )
            with self.assertRaises(ValueError):
                load_precomputed_dataset(path)
            loaded = load_precomputed_dataset(path, allow_legacy_pickle=True)
        self.assertEqual(loaded.metadata["dataset"], "toy")


if __name__ == "__main__":
    unittest.main()
