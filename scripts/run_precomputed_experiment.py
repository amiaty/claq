#!/usr/bin/env python3
"""Train and evaluate CLAQ from precomputed hard query responses.

This is the canonical paper-facing runner. It fits all information scores from
training data, selects checkpoints on validation data, evaluates each selected
checkpoint once on the held-out test split, and fits an independent
S|(H_T,Y) leakage probe with validation-selected regularization.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import claq
from claq.constraints import ConstraintConfig, build_static_proxy_admissible_mask
from claq.evaluation import aggregate_seed_metrics, fit_conditional_leakage_probe
from claq.precomputed import (
    build_mi_scorer_from_split,
    collect_tensor_rollouts,
    load_precomputed_dataset,
)
from claq.training import (
    ClaqObjectiveConfig,
    HistorySamplingConfig,
    build_claq_models,
    fit_claq_tensor,
    seed_everything,
)
from claq.utils import resolve_device, sha256_file


def parse_float_list(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid comma-separated float list: {value}") from exc


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid comma-separated integer list: {value}") from exc


def finite_or_none(value: float | None) -> float | None:
    if value is None or math.isinf(value):
        return None
    return float(value)


def jsonable(value: Any):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def cpu_state_dict(module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "run"


def validate_args(args: argparse.Namespace, *, num_queries: int) -> None:
    positive_ints = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "budget": args.budget,
        "mi_min_support": args.mi_min_support,
        "probe_max_iter": args.probe_max_iter,
    }
    for name, value in positive_ints.items():
        if value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    if args.learning_rate <= 0 or not math.isfinite(args.learning_rate):
        raise ValueError("--learning-rate must be finite and positive")
    if args.actor_temperature <= 0 or args.actor_temperature_end <= 0:
        raise ValueError("actor temperatures must be positive")
    if args.mi_alpha < 0 or not math.isfinite(args.mi_alpha):
        raise ValueError("--mi-alpha must be finite and nonnegative")
    for name in ("lambda_cost", "lambda_query_set", "lambda_fairness", "fairness_dual_lr"):
        value = float(getattr(args, name))
        if value < 0 or not math.isfinite(value):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative")
    for name in ("cost_budget", "leakage_budget", "fairness_tolerance", "fairness_lambda_max"):
        value = getattr(args, name)
        if value is not None and (value < 0 or not math.isfinite(value)):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if (args.proxy_leakage_threshold is None) != (args.proxy_label_threshold is None):
        raise ValueError("Both proxy thresholds must be supplied together")
    for name in ("proxy_leakage_threshold", "proxy_label_threshold"):
        value = getattr(args, name)
        if value is not None and (value < 0 or not math.isfinite(value)):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.dynamic_proxy_screen and args.proxy_leakage_threshold is None:
        raise ValueError("--dynamic-proxy-screen requires both proxy thresholds")
    if args.confidence_threshold is not None and not 0.0 < args.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must lie in (0,1]")
    effective_steps = min(args.budget, num_queries)
    if not 0 <= args.minimum_rollout_steps <= effective_steps:
        raise ValueError("--minimum-rollout-steps must lie in [0, min(budget, num_queries)]")
    if args.fairness_min_group_count < 1 or args.eo_min_group_count < 1:
        raise ValueError("fairness and EO minimum group counts must be at least 1")
    if args.fairness_tolerance is not None and args.fairness_dual_lr <= 0:
        raise ValueError("--fairness-tolerance requires a positive --fairness-dual-lr")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Canonical precomputed .npz dataset")
    parser.add_argument(
        "--allow-legacy-pickle",
        action="store_true",
        help="Open a trusted legacy archive containing NumPy object arrays",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete a nonempty output directory before starting",
    )
    parser.add_argument("--variant-name", default="CLAQ", help="Label stored in manifests and result rows")
    parser.add_argument("--lambdas", default="0,0.2,0.4", help="Comma-separated sensitive-adversary coefficients")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--budget", type=int, default=20, help="Maximum executed queries")
    parser.add_argument("--actor-temperature", type=float, default=1.0)
    parser.add_argument("--actor-temperature-end", type=float, default=0.2)
    parser.add_argument("--hidden-dims", default="2000,500", help="Two comma-separated MLP hidden dimensions")

    parser.add_argument("--lambda-cost", type=float, default=0.0)
    parser.add_argument("--lambda-query-set", type=float, default=0.0)
    parser.add_argument("--lambda-fairness", type=float, default=0.0)
    parser.add_argument("--fairness-tolerance", type=float, default=None)
    parser.add_argument("--fairness-dual-lr", type=float, default=0.0)
    parser.add_argument("--fairness-lambda-max", type=float, default=None)
    parser.add_argument("--fairness-min-group-count", type=int, default=2)
    parser.add_argument("--fairness-reduction", choices=("max", "mean"), default="max")

    parser.add_argument("--cost-budget", type=float, default=None)
    parser.add_argument("--disable-auto-unit-cost-budget", action="store_true")
    parser.add_argument("--leakage-budget", type=float, default=None)
    parser.add_argument("--proxy-leakage-threshold", type=float, default=None)
    parser.add_argument("--proxy-label-threshold", type=float, default=None)
    parser.add_argument("--dynamic-proxy-screen", action="store_true")
    parser.add_argument("--mi-alpha", type=float, default=0.5)
    parser.add_argument("--mi-min-support", type=int, default=32)

    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--minimum-rollout-steps", type=int, default=0)
    parser.add_argument("--selection-metric", default="acc")
    parser.add_argument("--selection-mode", choices=("max", "min"), default="max")
    parser.add_argument("--probe-c-grid", default="0.01,0.1,1,10")
    parser.add_argument("--probe-max-iter", type=int, default=2000)
    parser.add_argument("--report-equalized-odds", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--eo-min-group-count", type=int, default=1)

    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    device = resolve_device(args.device)
    dataset = load_precomputed_dataset(
        args.data,
        allow_legacy_pickle=args.allow_legacy_pickle,
    )
    validate_args(args, num_queries=dataset.num_queries)

    hidden_dims = parse_int_list(args.hidden_dims)
    if len(hidden_dims) != 2 or any(value < 1 for value in hidden_dims):
        raise ValueError("--hidden-dims must contain exactly two positive integers")
    lambdas = parse_float_list(args.lambdas)
    seeds = parse_int_list(args.seeds)
    probe_c_grid = parse_float_list(args.probe_c_grid)
    if not lambdas or not seeds or not probe_c_grid:
        raise ValueError("At least one lambda, seed, and probe C value are required")
    if any(value < 0 or not math.isfinite(value) for value in lambdas):
        raise ValueError("All sensitive-adversary coefficients must be finite and nonnegative")
    if any(value <= 0 or not math.isfinite(value) for value in probe_c_grid):
        raise ValueError("All probe C values must be finite and positive")

    if args.report_equalized_odds == "auto":
        report_equalized_odds = bool(dataset.metadata.get("report_equalized_odds", True))
    else:
        report_equalized_odds = args.report_equalized_odds == "true"

    if args.output.exists() and any(args.output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {args.output}. Use --overwrite to replace it."
            )
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checkpoints").mkdir(exist_ok=True)
    (args.output / "rollouts").mkdir(exist_ok=True)
    data_hash = sha256_file(args.data)
    max_steps = min(args.budget, dataset.num_queries)

    scorer = build_mi_scorer_from_split(
        dataset.train,
        num_label_classes=dataset.num_label_classes,
        num_sensitive_classes=dataset.num_sensitive_classes,
        alpha=args.mi_alpha,
        min_support=args.mi_min_support,
    )

    static_admissible = dataset.admissible_mask.clone().to(torch.bool)
    if args.proxy_leakage_threshold is not None:
        proxy_admissible = build_static_proxy_admissible_mask(
            scorer,
            leakage_threshold=args.proxy_leakage_threshold,
            label_information_threshold=args.proxy_label_threshold,
            justification_mask=dataset.justification_mask,
        )
        static_admissible &= proxy_admissible
    if not bool(static_admissible.any().item()):
        raise ValueError("The static admissibility and proxy masks exclude every query")

    effective_cost_budget = args.cost_budget
    if (
        effective_cost_budget is None
        and not args.disable_auto_unit_cost_budget
        and bool(torch.all(dataset.query_costs == 1).item())
    ):
        effective_cost_budget = float(max_steps)

    environment = {
        "claq_version": claq.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    manifest: dict[str, Any] = {
        "protocol": "canonical-precomputed-v1",
        "data": str(args.data.resolve()),
        "data_sha256": data_hash,
        "dataset_metadata": dataset.metadata,
        "dataset": {
            "num_queries": dataset.num_queries,
            "num_label_classes": dataset.num_label_classes,
            "num_sensitive_classes": dataset.num_sensitive_classes,
            "split_sizes": {
                "train": int(dataset.train.labels.numel()),
                "validation": int(dataset.validation.labels.numel()),
                "test": int(dataset.test.labels.numel()),
            },
            "query_names": list(dataset.query_names),
            "query_costs": dataset.query_costs.tolist(),
            "query_set_mask": dataset.query_set_mask.tolist(),
            "admissible_mask": dataset.admissible_mask.tolist(),
            "justification_mask": dataset.justification_mask.tolist(),
            "effective_static_admissible_mask": static_admissible.tolist(),
        },
        "arguments": vars(args),
        "derived": {
            "max_steps": max_steps,
            "effective_cost_budget": effective_cost_budget,
            "report_equalized_odds": report_equalized_odds,
        },
        "environment": environment,
    }
    with (args.output / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(jsonable(manifest), handle, indent=2)

    run_rows: list[dict[str, Any]] = []
    variant_name = safe_name(args.variant_name)
    for lambda_sensitive in lambdas:
        for seed in seeds:
            seed_everything(seed, deterministic=args.deterministic)
            train_loader, validation_loader, _ = dataset.loaders(
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=seed,
            )
            constraints = ConstraintConfig(
                query_costs=dataset.query_costs.clone(),
                cost_budget=finite_or_none(effective_cost_budget),
                leakage_budget=finite_or_none(args.leakage_budget),
                admissible_mask=static_admissible.clone(),
                justification_mask=dataset.justification_mask.clone(),
                proxy_leakage_threshold=args.proxy_leakage_threshold,
                proxy_label_threshold=args.proxy_label_threshold,
                dynamic_proxy_screen=bool(args.dynamic_proxy_screen),
            )
            objective = ClaqObjectiveConfig(
                lambda_sensitive=float(lambda_sensitive),
                lambda_cost=float(args.lambda_cost),
                lambda_query_set=float(args.lambda_query_set),
                lambda_fairness=float(args.lambda_fairness),
                fairness_tolerance=args.fairness_tolerance,
                fairness_dual_lr=float(args.fairness_dual_lr),
                fairness_lambda_max=args.fairness_lambda_max,
                fairness_min_group_count=args.fairness_min_group_count,
                fairness_reduction=args.fairness_reduction,
                query_set_mask=dataset.query_set_mask.clone(),
                condition_sensitive_on_y=True,
                num_sensitive_classes=dataset.num_sensitive_classes,
                constraints=constraints,
                mi_scorer=scorer,
                rollout_steps=max_steps,
                confidence_threshold=args.confidence_threshold,
                minimum_rollout_steps=args.minimum_rollout_steps,
                start_from_sampled_history=False,
            )
            objective.validate(dataset.num_queries)
            actor, classifier, sensitive_head = build_claq_models(
                max_queries=dataset.num_queries,
                num_classes=dataset.num_label_classes,
                device=device,
                actor_eps=args.actor_temperature,
                condition_sensitive_on_y=True,
                num_sensitive_classes=dataset.num_sensitive_classes,
                hidden_dims=hidden_dims,
            )
            optimizer = torch.optim.Adam(
                list(actor.parameters()) + list(classifier.parameters()) + list(sensitive_head.parameters()),
                lr=args.learning_rate,
            )
            history, best = fit_claq_tensor(
                actor=actor,
                classifier=classifier,
                s_head=sensitive_head,
                optimizer=optimizer,
                train_loader=train_loader,
                validation_loader=validation_loader,
                objective_config=objective,
                history_config=HistorySamplingConfig(min_history=0, max_history=0, non_sensitive_only=False),
                device=device,
                num_epochs=args.epochs,
                actor_eps_end=args.actor_temperature_end,
                selection_metric=args.selection_metric,
                selection_mode=args.selection_mode,
            )
            actor.load_state_dict(best["actor_state_dict"])
            classifier.load_state_dict(best["classifier_state_dict"])
            sensitive_head.load_state_dict(best["s_head_state_dict"])
            actor.change_eps(float(best["actor_eps"]))
            actor.eval()
            classifier.eval()
            sensitive_head.eval()

            rollout_kwargs = dict(
                actor=actor,
                classifier=classifier,
                device=device,
                max_steps=max_steps,
                batch_size=args.batch_size,
                constraint_config=constraints,
                mi_scorer=scorer,
                confidence_threshold=args.confidence_threshold,
                minimum_steps=args.minimum_rollout_steps,
                query_set_mask=dataset.query_set_mask,
                report_equalized_odds=report_equalized_odds,
                equalized_odds_min_group_count=args.eo_min_group_count,
                num_sensitive_classes=dataset.num_sensitive_classes,
            )
            # Train and validation rollouts fit/select the independent probe.
            train_rollout = collect_tensor_rollouts(split=dataset.train, **rollout_kwargs)
            validation_rollout = collect_tensor_rollouts(split=dataset.validation, **rollout_kwargs)
            # The held-out test split is evaluated once after checkpoint selection.
            test_rollout = collect_tensor_rollouts(split=dataset.test, **rollout_kwargs)

            probe = fit_conditional_leakage_probe(
                train_transcripts=train_rollout["transcripts"].numpy(),
                train_labels=train_rollout["labels"].numpy(),
                train_sensitive=train_rollout["sensitive"].numpy(),
                validation_transcripts=validation_rollout["transcripts"].numpy(),
                validation_labels=validation_rollout["labels"].numpy(),
                validation_sensitive=validation_rollout["sensitive"].numpy(),
                test_transcripts=test_rollout["transcripts"].numpy(),
                test_labels=test_rollout["labels"].numpy(),
                test_sensitive=test_rollout["sensitive"].numpy(),
                c_grid=probe_c_grid,
                max_iter=args.probe_max_iter,
                random_state=seed,
            )

            run_name = f"{variant_name}_lambda_{lambda_sensitive:g}_seed_{seed}"
            checkpoint_path = args.output / "checkpoints" / f"{run_name}.pt"
            checkpoint = {
                "format_version": 1,
                "claq_version": claq.__version__,
                "actor_state_dict": cpu_state_dict(actor),
                "classifier_state_dict": cpu_state_dict(classifier),
                "s_head_state_dict": cpu_state_dict(sensitive_head),
                "best_epoch": int(best["epoch"]),
                "validation_score": float(best["validation_score"]),
                "variant_name": args.variant_name,
                "seed": int(seed),
                "data_sha256": data_hash,
                "model_config": {
                    "num_queries": dataset.num_queries,
                    "num_label_classes": dataset.num_label_classes,
                    "num_sensitive_classes": dataset.num_sensitive_classes,
                    "condition_sensitive_on_y": True,
                    "hidden_dims": hidden_dims,
                    "actor_temperature_start": args.actor_temperature,
                    "actor_temperature_end": args.actor_temperature_end,
                    "actor_temperature_selected": float(best["actor_eps"]),
                },
                "objective_config": {
                    "lambda_sensitive": float(lambda_sensitive),
                    "lambda_cost": float(args.lambda_cost),
                    "lambda_query_set": float(args.lambda_query_set),
                    "lambda_fairness_initial": float(args.lambda_fairness),
                    "lambda_fairness_selected": float(best["lambda_fairness"]),
                    "lambda_fairness_final": float(objective.lambda_fairness),
                    "fairness_tolerance": args.fairness_tolerance,
                    "rollout_steps": max_steps,
                    "confidence_threshold": args.confidence_threshold,
                    "minimum_rollout_steps": args.minimum_rollout_steps,
                },
                "constraint_config": {
                    "cost_budget": finite_or_none(effective_cost_budget),
                    "leakage_budget": finite_or_none(args.leakage_budget),
                    "proxy_leakage_threshold": args.proxy_leakage_threshold,
                    "proxy_label_threshold": args.proxy_label_threshold,
                    "dynamic_proxy_screen": bool(args.dynamic_proxy_screen),
                    "query_costs": dataset.query_costs.tolist(),
                    "effective_static_admissible_mask": static_admissible.tolist(),
                    "justification_mask": dataset.justification_mask.tolist(),
                },
                "query_names": list(dataset.query_names),
                "dataset_metadata": dataset.metadata,
            }
            torch.save(checkpoint, checkpoint_path)
            np.savez_compressed(
                args.output / "rollouts" / f"{run_name}_test.npz",
                transcripts=test_rollout["transcripts"].numpy(),
                labels=test_rollout["labels"].numpy(),
                sensitive=test_rollout["sensitive"].numpy(),
                predictions=test_rollout["predictions"].numpy(),
                actions=test_rollout["actions"].numpy(),
                query_counts=test_rollout["query_counts"].numpy(),
                cumulative_cost=test_rollout["cumulative_cost"].numpy(),
                cumulative_leakage=test_rollout["cumulative_leakage"].numpy(),
                stop_reason=np.asarray(test_rollout["stop_reason"], dtype=object),
            )
            row = {
                "run_name": run_name,
                "variant_name": args.variant_name,
                "lambda_sensitive": float(lambda_sensitive),
                "lambda_fairness_selected": float(best["lambda_fairness"]),
                "lambda_fairness_final": float(objective.lambda_fairness),
                "seed": int(seed),
                "best_epoch": int(best["epoch"]),
                "validation_score": float(best["validation_score"]),
                **test_rollout["metrics"],
                "probe_conditional_entropy_baseline_bits": probe.conditional_entropy_baseline_bits,
                "probe_cross_entropy_bits": probe.probe_cross_entropy_bits,
                "probe_leakage_lower_bound_bits": probe.probe_leakage_lower_bound_bits,
                "probe_accuracy": probe.probe_accuracy,
                "checkpoint": str(checkpoint_path),
            }
            run_rows.append(row)
            with (args.output / f"{run_name}_history.json").open("w", encoding="utf-8") as handle:
                json.dump(jsonable(history), handle, indent=2)
            with (args.output / f"{run_name}_metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(jsonable(row), handle, indent=2)
            print(json.dumps(jsonable(row), indent=2))

    metric_names = [
        "accuracy",
        "macro_f1",
        "equalized_odds_violation",
        "equalized_odds_minimum_included_cell_count",
        "mean_queries",
        "mean_cost",
        "mean_empirical_leakage_account",
        "designated_query_rate",
        "no_feasible_query_rate",
        "confidence_rate",
        "max_steps_rate",
        "all_queries_exhausted_rate",
        "probe_leakage_lower_bound_bits",
        "probe_accuracy",
    ]
    summary: dict[str, Any] = {
        **manifest,
        "runs": run_rows,
        "by_lambda": {},
    }
    for lambda_sensitive in lambdas:
        rows = [row for row in run_rows if row["lambda_sensitive"] == float(lambda_sensitive)]
        available = [metric for metric in metric_names if all(metric in row for row in rows)]
        summary["by_lambda"][str(lambda_sensitive)] = aggregate_seed_metrics(rows, available)
    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(jsonable(summary), handle, indent=2)


if __name__ == "__main__":
    main()
