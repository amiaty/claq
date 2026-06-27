#!/usr/bin/env python3
"""Validate and summarize a canonical CLAQ precomputed-response archive."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from claq.precomputed import load_precomputed_dataset
from claq.utils import sha256_file


def distribution(values: torch.Tensor) -> dict[str, int]:
    return {str(int(key)): int(count) for key, count in sorted(Counter(values.tolist()).items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument(
        "--allow-legacy-pickle",
        action="store_true",
        help="Open a trusted legacy archive containing NumPy object arrays",
    )
    args = parser.parse_args()

    dataset = load_precomputed_dataset(
        args.data,
        allow_legacy_pickle=args.allow_legacy_pickle,
    )
    report = {
        "path": str(args.data.resolve()),
        "sha256": sha256_file(args.data),
        "num_queries": dataset.num_queries,
        "num_label_classes": dataset.num_label_classes,
        "num_sensitive_classes": dataset.num_sensitive_classes,
        "query_names_unique": len(set(dataset.query_names)) == len(dataset.query_names),
        "designated_query_count": int(dataset.query_set_mask.sum().item()),
        "admissible_query_count": int(dataset.admissible_mask.sum().item()),
        "justified_query_count": int(dataset.justification_mask.sum().item()),
        "query_cost_min": float(dataset.query_costs.min().item()),
        "query_cost_max": float(dataset.query_costs.max().item()),
        "splits": {},
        "metadata": dataset.metadata,
    }
    for name in ("train", "validation", "test"):
        split = getattr(dataset, name)
        report["splits"][name] = {
            "num_examples": int(split.labels.numel()),
            "label_distribution": distribution(split.labels),
            "sensitive_distribution": distribution(split.sensitive),
            "positive_response_rate": float((split.responses > 0).to(torch.float32).mean().item()),
        }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"VALID: {args.data}")
        print(f"SHA-256: {report['sha256']}")
        print(
            f"Queries: {dataset.num_queries} | labels: {dataset.num_label_classes} | "
            f"sensitive classes: {dataset.num_sensitive_classes}"
        )
        for name, values in report["splits"].items():
            print(
                f"{name:10s} N={values['num_examples']} "
                f"Y={values['label_distribution']} S={values['sensitive_distribution']}"
            )


if __name__ == "__main__":
    main()
