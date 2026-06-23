"""Tiny-start replay sampling for sample-level CLAQ intuition figures.

Each replay starts both questioners from an *empty* knowledge state and lets
them ask one query at a time until they reach confidence. We then keep the
samples that best illustrate the intended story: the baseline spends a
sensitive query that CLAQ avoids, while both still reach the correct class.
"""

from __future__ import annotations

import torch
import numpy as np
from tqdm.auto import tqdm

from claq.analysis.rollouts import first_divergence_step, rollout_until_confidence


def _rollout(bundle, answers_row, empty_mask, *, concepts, sensitive_mask, class_names,
             confidence_threshold, rollout_max_steps, positive_class_idx, positive_class_name):
    return rollout_until_confidence(
        bundle=bundle,
        answers_row=answers_row,
        init_mask=empty_mask,
        concepts=concepts,
        sensitive_mask=sensitive_mask,
        class_names=class_names,
        threshold=confidence_threshold,
        max_steps=rollout_max_steps,
        positive_class_idx=positive_class_idx,
        positive_class_name=positive_class_name,
    )


def _illustrativeness(row: dict, prefer_baseline_sensitive: bool) -> tuple:
    """Higher is a better intuition example. Sorted descending."""
    baseline = row["baseline"]
    claq = row["claq"]
    baseline_leaks = baseline["sensitive_steps"] > 0
    claq_is_cleaner = baseline["sensitive_steps"] > claq["sensitive_steps"]
    diverged = row["first_divergence_step"] is not None
    return (
        int(prefer_baseline_sensitive and baseline_leaks),  # baseline spends a sensitive query
        int(claq_is_cleaner),                               # CLAQ asks fewer of them
        int(row["both_correct"]),                           # neither is wrong
        int(diverged),                                      # the two paths actually differ
        row["sensitive_gap"],                               # prefer a larger gap
        -max(baseline["queries_asked"], claq["queries_asked"]),  # prefer shorter rollouts
    )


def _select_balanced(ranked: list[dict], num_cases: int, bucket_key) -> list[dict]:
    """Round-robin across buckets, keeping each bucket's ranked order."""
    buckets: dict = {}
    for row in ranked:
        buckets.setdefault(bucket_key(row), []).append(row)

    selected: list[dict] = []
    keys = sorted(buckets)
    while len(selected) < num_cases:
        progressed = False
        for key in keys:
            if buckets[key]:
                selected.append(buckets[key].pop(0))
                progressed = True
                if len(selected) >= num_cases:
                    break
        if not progressed:
            break
    return selected


def sample_intuition_replays(
    dataset,
    answer_builder,
    baseline_bundle: dict,
    claq_bundle: dict,
    concepts: list[str],
    sensitive_mask: torch.Tensor,
    class_names: list[str],
    *,
    num_cases: int = 4,
    pool_size: int = 400,
    random_seed: int = 0,
    confidence_threshold: float = 0.95,
    rollout_max_steps: int = 20,
    positive_class_idx: int | None = None,
    positive_class_name: str | None = None,
    prefer_baseline_sensitive: bool = True,
    balance_labels: bool = False,
    balance_concept_idx: int | None = None,
    balance_concept_name: str | None = None,
) -> list[dict]:
    """Roll out baseline vs CLAQ from an empty start and return the top cases.

    Both questioners see the same concept answers and start from zero knowledge,
    so the only difference is the actor. Returns ``num_cases`` records ready for
    ``plot_rollout_comparisons``.
    """
    rng = np.random.default_rng(random_seed)
    sample_indices = rng.permutation(len(dataset))[: min(pool_size, len(dataset))]
    sensitive_indices = (sensitive_mask > 0.5).nonzero(as_tuple=False).flatten().cpu()

    records: list[dict] = []
    with torch.no_grad():
        for sample_idx in tqdm(sample_indices, desc="Sampling intuition replays"):
            sample_idx = int(sample_idx)
            image, label_idx = dataset[sample_idx]
            label_idx = int(label_idx)

            answers = answer_builder(image.unsqueeze(0))
            answers_row = answers[0]
            empty_mask = torch.zeros_like(answers)

            rollout_kwargs = dict(
                concepts=concepts,
                sensitive_mask=sensitive_mask,
                class_names=class_names,
                confidence_threshold=confidence_threshold,
                rollout_max_steps=rollout_max_steps,
                positive_class_idx=positive_class_idx,
                positive_class_name=positive_class_name,
            )
            baseline_stop = _rollout(baseline_bundle, answers_row, empty_mask, **rollout_kwargs)
            claq_stop = _rollout(claq_bundle, answers_row, empty_mask, **rollout_kwargs)

            # Skip the degenerate case where neither questioner asks anything.
            if baseline_stop["queries_asked"] == 0 and claq_stop["queries_asked"] == 0:
                continue

            record = {
                "sample_idx": sample_idx,
                "label_idx": label_idx,
                "label_name": class_names[label_idx],
                "initial_history": [],
                "initial_history_size": 0,
                "baseline": baseline_stop,
                "claq": claq_stop,
                "both_correct": bool(
                    baseline_stop["final_pred_idx"] == label_idx
                    and claq_stop["final_pred_idx"] == label_idx
                ),
                "sensitive_gap": baseline_stop["sensitive_steps"] - claq_stop["sensitive_steps"],
                "first_divergence_step": first_divergence_step(
                    baseline_stop["sequence"], claq_stop["sequence"]
                ),
            }

            if balance_concept_idx is not None:
                if hasattr(dataset, "query_targets"):
                    value = float(dataset.query_targets[sample_idx, balance_concept_idx].item())
                else:
                    value = float(answers_row[balance_concept_idx].item())
                record["balance_concept_idx"] = int(balance_concept_idx)
                record["balance_concept_name"] = balance_concept_name or concepts[balance_concept_idx]
                record["balance_concept_value"] = int(value > 0.5)

            records.append(record)

    if not records:
        return []

    ranked = sorted(
        records,
        key=lambda row: _illustrativeness(row, prefer_baseline_sensitive),
        reverse=True,
    )

    if not balance_labels:
        return ranked[:num_cases]

    def bucket_key(row: dict):
        if "balance_concept_value" in row:
            return (row["label_idx"], row["balance_concept_value"])
        return (row["label_idx"],)

    return _select_balanced(ranked, num_cases, bucket_key)
