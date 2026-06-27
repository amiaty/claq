from __future__ import annotations

import unittest

import numpy as np

from claq.evaluation import fit_conditional_leakage_probe


class ProbeTests(unittest.TestCase):
    def test_rejects_unseen_sensitive_class(self):
        with self.assertRaises(ValueError):
            fit_conditional_leakage_probe(
                train_transcripts=np.zeros((4, 2)),
                train_labels=np.array([0, 0, 1, 1]),
                train_sensitive=np.array([0, 1, 0, 1]),
                validation_transcripts=np.zeros((2, 2)),
                validation_labels=np.array([0, 1]),
                validation_sensitive=np.array([0, 1]),
                test_transcripts=np.zeros((2, 2)),
                test_labels=np.array([0, 1]),
                test_sensitive=np.array([0, 2]),
            )


if __name__ == "__main__":
    unittest.main()
