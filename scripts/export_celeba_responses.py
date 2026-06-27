#!/usr/bin/env python3
"""Export directly observed CelebA attributes to the canonical CLAQ .npz format."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from claq.data import get_celeba_datasets, load_celeba_attribute_spec
from claq.precomputed import PrecomputedDataset, PrecomputedSplit, save_precomputed_dataset


DEFAULT_ASSOCIATED = (
    "Male", "No_Beard", "Mustache", "Goatee", "Sideburns",
    "5_o_Clock_Shadow", "Heavy_Makeup", "Wearing_Lipstick",
)


def split_from_dataset(dataset, sensitive_query_index: int) -> PrecomputedSplit:
    responses_01 = dataset.query_targets.to(torch.float32)
    responses = responses_01 * 2.0 - 1.0
    labels = dataset.base_dataset.attr[:, dataset.spec.target_index].to(torch.long)
    sensitive = dataset.query_targets[:, sensitive_query_index].to(torch.long)
    return PrecomputedSplit(responses=responses, labels=labels, sensitive=sensitive)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Torchvision CelebA root")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", default="Attractive")
    parser.add_argument("--sensitive", default="Male")
    parser.add_argument("--associated", default=",".join(DEFAULT_ASSOCIATED),
                        help="Comma-separated query-composition set")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    associated = tuple(item.strip() for item in args.associated.split(",") if item.strip())
    if not associated:
        raise ValueError("--associated must contain at least one query name")
    if len(set(associated)) != len(associated):
        raise ValueError("--associated contains duplicate query names")
    spec = load_celeba_attribute_spec(
        root=args.root,
        target_attribute=args.target,
        sensitive_attributes=associated,
        download=args.download,
    )
    if args.sensitive not in spec.query_attribute_names:
        raise ValueError("The sensitive attribute must remain in the query vocabulary")
    sensitive_query_index = spec.query_attribute_names.index(args.sensitive)
    train_ds, validation_ds, test_ds = get_celeba_datasets(
        transform=None,
        root=args.root,
        spec=spec,
        return_query_targets=True,
        download=False,
    )
    query_set_mask = torch.tensor(
        [name in set(associated) for name in spec.query_attribute_names],
        dtype=torch.bool,
    )
    dataset = PrecomputedDataset(
        train=split_from_dataset(train_ds, sensitive_query_index),
        validation=split_from_dataset(validation_ds, sensitive_query_index),
        test=split_from_dataset(test_ds, sensitive_query_index),
        query_names=tuple(spec.query_attribute_names),
        query_set_mask=query_set_mask,
        query_costs=torch.ones(spec.num_queries),
        admissible_mask=torch.ones(spec.num_queries, dtype=torch.bool),
        justification_mask=torch.zeros(spec.num_queries, dtype=torch.bool),
        metadata={
            "dataset": "CelebA",
            "target_attribute": args.target,
            "sensitive_attribute": args.sensitive,
            "response_source": "dataset-provided attribute labels",
            "query_composition_set": list(associated),
            "operational_response_alphabet": [-1, 1],
            "report_equalized_odds": True,
            "split_sizes": {
                "train": len(train_ds),
                "validation": len(validation_ds),
                "test": len(test_ds),
            },
        },
    )
    print(save_precomputed_dataset(args.output, dataset))


if __name__ == "__main__":
    main()
