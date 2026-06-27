#!/usr/bin/env python3
"""Run the six paper-defined CLAQ ablation variants with a common protocol."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from claq.precomputed import load_precomputed_dataset


VARIANTS = (
    "CLAQ-Base",
    "CLAQ-Cost",
    "CLAQ-Leak",
    "CLAQ-LeakCost",
    "CLAQ-Fair",
    "CLAQ-Full",
)


def append_option(command: list[str], name: str, value) -> None:
    if value is not None:
        command.extend([name, str(value)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--allow-legacy-pickle", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--hidden-dims", default="2000,500")
    parser.add_argument("--actor-temperature", type=float, default=1.0)
    parser.add_argument("--actor-temperature-end", type=float, default=0.2)
    parser.add_argument("--lambda-sensitive", type=float, default=0.4)
    parser.add_argument("--lambda-cost", type=float, default=0.1)
    parser.add_argument("--lambda-fairness", type=float, default=0.1)
    parser.add_argument("--lambda-query-set", type=float, default=0.1)
    parser.add_argument("--cost-budget", type=float, default=None)
    parser.add_argument("--leakage-budget", type=float, default=0.25)
    parser.add_argument("--proxy-leakage-threshold", type=float, default=None)
    parser.add_argument("--proxy-label-threshold", type=float, default=None)
    parser.add_argument("--fairness-tolerance", type=float, default=None)
    parser.add_argument("--fairness-dual-lr", type=float, default=0.0)
    parser.add_argument("--selection-metric", default="acc")
    parser.add_argument("--selection-mode", choices=("max", "min"), default="max")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--allow-degenerate-unit-cost-ablation",
        action="store_true",
        help="Permit cost variants when unit cost and the horizon make the cost component behaviorally redundant",
    )
    args = parser.parse_args()

    requested = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    unknown = sorted(set(requested) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Supported: {list(VARIANTS)}")
    if not requested:
        raise ValueError("At least one variant is required")
    if (args.proxy_leakage_threshold is None) != (args.proxy_label_threshold is None):
        raise ValueError("Both proxy thresholds must be supplied together")
    if "CLAQ-Full" in requested and args.proxy_leakage_threshold is None:
        raise ValueError("CLAQ-Full requires both proxy thresholds so that dynamic proxy screening is active")
    if args.fairness_tolerance is not None and args.fairness_dual_lr <= 0:
        raise ValueError("--fairness-tolerance requires a positive --fairness-dual-lr")

    dataset = load_precomputed_dataset(
        args.data,
        allow_legacy_pickle=args.allow_legacy_pickle,
    )
    cost_variants = {"CLAQ-Cost", "CLAQ-LeakCost", "CLAQ-Fair", "CLAQ-Full"}
    if set(requested) & cost_variants and bool(torch.all(dataset.query_costs == dataset.query_costs[0]).item()):
        budget_is_redundant = args.cost_budget is None or args.cost_budget >= args.budget * float(dataset.query_costs[0])
        if budget_is_redundant and not args.allow_degenerate_unit_cost_ablation:
            raise ValueError(
                "The requested cost ablation is behaviorally degenerate: all query costs are equal, "
                "the cost budget does not bind before the horizon, and the cost penalty cannot rank candidates. "
                "Provide heterogeneous query_costs, set a binding --cost-budget, or explicitly pass "
                "--allow-degenerate-unit-cost-ablation for a diagnostic run."
            )

    if args.output.exists() and any(args.output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {args.output}. Use --overwrite to replace it."
            )
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    combined: dict[str, object] = {"data": str(args.data), "variants": {}}
    runner = Path(__file__).with_name("run_precomputed_experiment.py")

    for variant in requested:
        output = args.output / variant
        command = [
            sys.executable,
            str(runner),
            "--data", str(args.data),
            "--output", str(output),
            "--variant-name", variant,
            "--lambdas", "0" if variant in {"CLAQ-Base", "CLAQ-Cost"} else str(args.lambda_sensitive),
            "--seeds", args.seeds,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate),
            "--budget", str(args.budget),
            "--hidden-dims", args.hidden_dims,
            "--actor-temperature", str(args.actor_temperature),
            "--actor-temperature-end", str(args.actor_temperature_end),
            "--selection-metric", args.selection_metric,
            "--selection-mode", args.selection_mode,
            "--device", args.device,
            "--num-workers", str(args.num_workers),
        ]
        if args.deterministic:
            command.append("--deterministic")
        if args.allow_legacy_pickle:
            command.append("--allow-legacy-pickle")
        if args.proxy_leakage_threshold is not None:
            command.extend([
                "--proxy-leakage-threshold", str(args.proxy_leakage_threshold),
                "--proxy-label-threshold", str(args.proxy_label_threshold),
            ])
        if variant in {"CLAQ-Base", "CLAQ-Leak"}:
            command.append("--disable-auto-unit-cost-budget")
        if variant in {"CLAQ-Cost", "CLAQ-LeakCost", "CLAQ-Fair", "CLAQ-Full"}:
            command.extend(["--lambda-cost", str(args.lambda_cost)])
            append_option(command, "--cost-budget", args.cost_budget)
        if variant in {"CLAQ-Leak", "CLAQ-LeakCost", "CLAQ-Fair", "CLAQ-Full"}:
            append_option(command, "--leakage-budget", args.leakage_budget)
        if variant in {"CLAQ-Fair", "CLAQ-Full"}:
            command.extend(["--lambda-fairness", str(args.lambda_fairness)])
            append_option(command, "--fairness-tolerance", args.fairness_tolerance)
            if args.fairness_tolerance is not None:
                command.extend(["--fairness-dual-lr", str(args.fairness_dual_lr)])
        if variant == "CLAQ-Full":
            command.extend(["--lambda-query-set", str(args.lambda_query_set)])
            if args.proxy_leakage_threshold is not None:
                command.append("--dynamic-proxy-screen")

        print("RUN:", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        with (output / "summary.json").open("r", encoding="utf-8") as handle:
            combined["variants"][variant] = json.load(handle)

    with (args.output / "ablation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(combined, handle, indent=2)


if __name__ == "__main__":
    main()
