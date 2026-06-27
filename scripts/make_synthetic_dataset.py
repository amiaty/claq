#!/usr/bin/env python3
"""Create a small deterministic CLAQ archive for end-to-end verification."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from claq.precomputed import PrecomputedDataset, PrecomputedSplit, save_precomputed_dataset


def _split(rng: np.random.Generator, n: int, num_queries: int) -> PrecomputedSplit:
    if n < 8:
        raise ValueError("Each split must contain at least eight examples")
    y = rng.integers(0, 2, size=n, dtype=np.int64)
    # Keep S correlated with Y without making the variables identical.
    s = np.bitwise_xor(y, rng.binomial(1, 0.25, size=n).astype(np.int64))
    responses = np.empty((n, num_queries), dtype=np.int64)

    for q in range(0, min(4, num_queries)):
        noise = rng.binomial(1, 0.10 + 0.03 * q, size=n).astype(np.int64)
        responses[:, q] = np.bitwise_xor(y, noise)

    for q in range(4, min(8, num_queries)):
        noise = rng.binomial(1, 0.08 + 0.03 * (q - 4), size=n).astype(np.int64)
        responses[:, q] = np.bitwise_xor(s, noise)

    for q in range(8, num_queries):
        if q % 2 == 0:
            base = np.bitwise_xor(y, s)
            noise_rate = 0.20
        else:
            base = rng.integers(0, 2, size=n, dtype=np.int64)
            noise_rate = 0.0
        noise = rng.binomial(1, noise_rate, size=n).astype(np.int64)
        responses[:, q] = np.bitwise_xor(base, noise)

    responses = responses * 2 - 1
    return PrecomputedSplit(
        responses=torch.as_tensor(responses, dtype=torch.float32),
        labels=torch.as_tensor(y, dtype=torch.long),
        sensitive=torch.as_tensor(s, dtype=torch.long),
    )


def create_synthetic_dataset(
    output: Path,
    *,
    seed: int = 7,
    num_queries: int = 8,
    train_size: int = 64,
    validation_size: int = 32,
    test_size: int = 32,
) -> Path:
    """Create and save a deterministic software-verification archive."""

    if num_queries < 8:
        raise ValueError("num_queries must be at least 8")
    rng = np.random.default_rng(seed)
    train = _split(rng, train_size, num_queries)
    validation = _split(rng, validation_size, num_queries)
    test = _split(rng, test_size, num_queries)

    query_names = tuple(f"q_{index:02d}" for index in range(num_queries))
    query_set_mask = torch.zeros(num_queries, dtype=torch.bool)
    query_set_mask[4:min(8, num_queries)] = True
    dataset = PrecomputedDataset(
        train=train,
        validation=validation,
        test=test,
        query_names=query_names,
        query_set_mask=query_set_mask,
        query_costs=torch.ones(num_queries, dtype=torch.float32),
        admissible_mask=torch.ones(num_queries, dtype=torch.bool),
        justification_mask=torch.zeros(num_queries, dtype=torch.bool),
        metadata={
            "dataset": "synthetic_claq_quickstart",
            "seed": seed,
            "target": "binary utility label",
            "sensitive": "binary correlated attribute",
            "response_source": "deterministic synthetic generator",
            "operational_response_alphabet": [-1, 1],
            "report_equalized_odds": True,
            "purpose": "software verification only; not a scientific benchmark",
        },
    )
    return save_precomputed_dataset(output, dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/precomputed/synthetic.npz"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=64)
    parser.add_argument("--validation-size", type=int, default=32)
    parser.add_argument("--test-size", type=int, default=32)
    args = parser.parse_args()
    path = create_synthetic_dataset(
        args.output,
        seed=args.seed,
        num_queries=args.num_queries,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
    )
    print(path)


if __name__ == "__main__":
    main()
