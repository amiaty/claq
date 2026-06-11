"""Deterministic fixed-history evaluation for comparing saved runs."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from claq.analysis.rollouts import build_random_initial_history
from claq.core.runtime import apply_query_distribution


def evaluate_bundles_on_fixed_histories(
    dataset,
    answer_builder,
    bundles_by_name: dict[str, dict],
    sensitive_mask: torch.Tensor,
    *,
    min_history: int = 1,
    max_history: int = 2,
    history_mode: str = "non_sensitive",
    num_trials: int = 8,
    max_samples: int | None = None,
    eval_seed: int = 0,
    desc: str = "Fixed-history reevaluation",
    positive_class_idx: int | None = None,
    positive_class_name: str | None = None,
) -> list[dict]:
    if not bundles_by_name:
        raise ValueError("bundles_by_name must not be empty")
    if num_trials < 1:
        raise ValueError("num_trials must be at least 1")

    sample_indices = np.arange(len(dataset))
    if max_samples is not None:
        rng = np.random.default_rng(eval_seed)
        sample_indices = rng.permutation(len(dataset))[: min(max_samples, len(dataset))]

    sensitive_indices = (sensitive_mask > 0.5).nonzero(as_tuple=False).flatten().cpu()
    bundle_names = list(bundles_by_name.keys())
    trial_stats = {
        name: [
            {
                "correct": 0,
                "total": 0,
                "sensitive_query_rate": 0.0,
                "avg_confidence": 0.0,
                "avg_positive_prob": 0.0,
            }
            for _ in range(num_trials)
        ]
        for name in bundle_names
    }

    with torch.no_grad():
        for sample_idx in tqdm(sample_indices, desc=desc):
            image, label_idx = dataset[int(sample_idx)]
            answers = answer_builder(image.unsqueeze(0))
            label_idx = int(label_idx)

            for trial in range(num_trials):
                init_mask, masked_answers, _ = build_random_initial_history(
                    answers=answers,
                    sample_idx=int(sample_idx),
                    trial=trial,
                    min_history=min_history,
                    max_history=max_history,
                    mode=history_mode,
                    sensitive_indices=sensitive_indices,
                )

                for name, bundle in bundles_by_name.items():
                    query_distribution = bundle["actor"](masked_answers, init_mask)
                    updated_answers = apply_query_distribution(
                        masked_answers=masked_answers,
                        answers=answers,
                        query_distribution=query_distribution,
                    )
                    logits = bundle["classifier"](updated_answers)
                    probs = F.softmax(logits, dim=1)
                    conf, pred = probs.max(dim=1)
                    sensitive_mask_for_bundle = sensitive_mask.to(query_distribution.device).unsqueeze(0)

                    stats = trial_stats[name][trial]
                    stats["correct"] += int(pred[0].item() == label_idx)
                    stats["total"] += 1
                    stats["sensitive_query_rate"] += float(
                        (query_distribution * sensitive_mask_for_bundle).sum(dim=1).item()
                    )
                    stats["avg_confidence"] += float(conf[0].item())
                    if positive_class_idx is not None:
                        stats["avg_positive_prob"] += float(probs[0, positive_class_idx].item())

    summary_rows = []
    for name, bundle in bundles_by_name.items():
        meta = bundle.get("meta", {})
        trial_rows = []
        for trial, stats in enumerate(trial_stats[name]):
            total = max(stats["total"], 1)
            trial_rows.append(
                {
                    "trial": int(trial),
                    "acc": float(stats["correct"] / total),
                    "sensitive_query_rate": float(stats["sensitive_query_rate"] / total),
                    "avg_confidence": float(stats["avg_confidence"] / total),
                    "avg_positive_prob": None
                    if positive_class_idx is None
                    else float(stats["avg_positive_prob"] / total),
                }
            )

        acc_values = np.array([row["acc"] for row in trial_rows], dtype=float)
        sens_values = np.array([row["sensitive_query_rate"] for row in trial_rows], dtype=float)
        conf_values = np.array([row["avg_confidence"] for row in trial_rows], dtype=float)
        positive_prob_values = None
        if positive_class_idx is not None:
            positive_prob_values = np.array([row["avg_positive_prob"] for row in trial_rows], dtype=float)
        lambda_s = meta.get("lambda_s")
        lambda_c = meta.get("lambda_c")

        row = {
            "run_name": name,
            "lambda_s": None if lambda_s is None else float(lambda_s),
            "lambda_c": None if lambda_c is None else float(lambda_c),
            "num_samples": int(len(sample_indices)),
            "num_trials": int(num_trials),
            "history_mode": history_mode,
            "history_range": [int(min_history), int(max_history)],
            "mean_acc": float(acc_values.mean()),
            "std_acc": float(acc_values.std(ddof=0)),
            "mean_sensitive_query_rate": float(sens_values.mean()),
            "std_sensitive_query_rate": float(sens_values.std(ddof=0)),
            "mean_confidence": float(conf_values.mean()),
            "std_confidence": float(conf_values.std(ddof=0)),
            "trial_rows": trial_rows,
        }
        if positive_class_idx is not None and positive_prob_values is not None:
            row["positive_class_idx"] = int(positive_class_idx)
            row["positive_class_name"] = positive_class_name
            row["mean_positive_prob"] = float(positive_prob_values.mean())
            row["std_positive_prob"] = float(positive_prob_values.std(ddof=0))
        summary_rows.append(row)

    return sorted(
        summary_rows,
        key=lambda row: (
            float("-inf") if row["lambda_s"] is None else row["lambda_s"],
            row["run_name"],
        ),
    )
