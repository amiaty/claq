"""Replay sampling helpers for sample-level CLAQ analysis."""

from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm

from claq.analysis.rollouts import (
    build_random_initial_history,
    first_divergence_step,
    rollout_until_confidence,
)


def format_history(history_idx: list[int], answers_row: torch.Tensor, concepts: list[str], max_items: int = 4) -> list[str]:
    parts = []
    for idx in history_idx[:max_items]:
        answer_name = "yes" if float(answers_row[idx].item()) > 0 else "no"
        parts.append(f"{concepts[idx]}={answer_name}")
    if len(history_idx) > max_items:
        parts.append("...")
    return parts


def _build_record(
    sample_idx: int,
    trial: int,
    label_idx: int,
    label_name: str,
    init_history_idx: list[int],
    answers_row: torch.Tensor,
    baseline_stop: dict,
    claq_stop: dict,
    concepts: list[str],
) -> dict:
    both_correct = bool(
        (baseline_stop["final_pred_idx"] == label_idx) and (claq_stop["final_pred_idx"] == label_idx)
    )
    return {
        "sample_idx": int(sample_idx),
        "trial": int(trial),
        "label_idx": int(label_idx),
        "label_name": label_name,
        "initial_history": format_history(init_history_idx, answers_row, concepts, max_items=max(1, len(init_history_idx))),
        "initial_history_size": int(len(init_history_idx)),
        "baseline": baseline_stop,
        "claq": claq_stop,
        "both_correct": both_correct,
        "sensitive_gap": int(baseline_stop["sensitive_steps"] - claq_stop["sensitive_steps"]),
        "first_divergence_step": first_divergence_step(baseline_stop["sequence"], claq_stop["sequence"]),
    }


def sample_intuition_replays(
    dataset,
    answer_builder,
    baseline_bundle: dict,
    claq_bundle: dict,
    concepts: list[str],
    sensitive_mask: torch.Tensor,
    class_names: list[str],
    num_cases: int = 8,
    pool_size: int = 400,
    num_trials: int = 2,
    random_seed: int = 0,
    min_history: int = 1,
    max_history: int = 2,
    history_mode: str = "non_sensitive",
    require_nontrivial: bool = True,
    prefer_divergent: bool = True,
    prefer_baseline_sensitive: bool = True,
    confidence_threshold: float = 0.95,
    rollout_max_steps: int = 20,
    positive_class_idx: int | None = None,
    positive_class_name: str | None = None,
    balance_labels: bool = False,
    balance_concept_idx: int | None = None,
    balance_concept_name: str | None = None,
) -> list[dict]:
    rng = np.random.default_rng(random_seed)
    sample_indices = rng.permutation(len(dataset))[: min(pool_size, len(dataset))]
    sensitive_indices = (sensitive_mask > 0.5).nonzero(as_tuple=False).flatten().cpu()
    records = []

    with torch.no_grad():
        for sample_idx in tqdm(sample_indices, desc="Sampling intuition replays"):
            image, label_idx = dataset[int(sample_idx)]
            answers = answer_builder(image.unsqueeze(0))
            answers_row = answers[0]
            label_idx = int(label_idx)
            label_name = class_names[label_idx]

            for trial in range(num_trials):
                init_mask, _, init_history_idx = build_random_initial_history(
                    answers=answers,
                    sample_idx=int(sample_idx),
                    trial=trial,
                    min_history=min_history,
                    max_history=max_history,
                    mode=history_mode,
                    sensitive_indices=sensitive_indices,
                )
                baseline_stop = rollout_until_confidence(
                    bundle=baseline_bundle,
                    answers_row=answers_row,
                    init_mask=init_mask,
                    concepts=concepts,
                    sensitive_mask=sensitive_mask,
                    class_names=class_names,
                    threshold=confidence_threshold,
                    max_steps=rollout_max_steps,
                    positive_class_idx=positive_class_idx,
                    positive_class_name=positive_class_name,
                )
                claq_stop = rollout_until_confidence(
                    bundle=claq_bundle,
                    answers_row=answers_row,
                    init_mask=init_mask,
                    concepts=concepts,
                    sensitive_mask=sensitive_mask,
                    class_names=class_names,
                    threshold=confidence_threshold,
                    max_steps=rollout_max_steps,
                    positive_class_idx=positive_class_idx,
                    positive_class_name=positive_class_name,
                )
                if require_nontrivial and baseline_stop["queries_asked"] == 0 and claq_stop["queries_asked"] == 0:
                    continue
                record = _build_record(
                    sample_idx=int(sample_idx),
                    trial=trial,
                    label_idx=label_idx,
                    label_name=label_name,
                    init_history_idx=init_history_idx,
                    answers_row=answers_row,
                    baseline_stop=baseline_stop,
                    claq_stop=claq_stop,
                    concepts=concepts,
                )
                if balance_concept_idx is not None and hasattr(dataset, "query_targets"):
                    concept_value = float(dataset.query_targets[int(sample_idx), int(balance_concept_idx)].item())
                    record["balance_concept_idx"] = int(balance_concept_idx)
                    record["balance_concept_name"] = balance_concept_name or concepts[int(balance_concept_idx)]
                    record["balance_concept_value"] = int(concept_value > 0.5)
                records.append(record)

    def _intuition_sort_key(row: dict):
        divergence = row["first_divergence_step"]
        baseline_sensitive = row["baseline"]["sensitive_steps"] > 0
        claq_avoids_sensitive = row["baseline"]["sensitive_steps"] > row["claq"]["sensitive_steps"]
        return (
            1 if (prefer_baseline_sensitive and baseline_sensitive) else 0,
            1 if claq_avoids_sensitive else 0,
            1 if row["both_correct"] else 0,
            1 if (prefer_divergent and divergence is not None) else 0,
            abs(row["baseline"]["queries_asked"] - row["claq"]["queries_asked"]),
            abs(row["sensitive_gap"]),
            row["baseline"]["sensitive_steps"],
            max(row["baseline"]["queries_asked"], row["claq"]["queries_asked"]),
            -(divergence if divergence is not None else 999),
        )

    baseline_sensitive_gap = [
        row for row in records if row["baseline"]["sensitive_steps"] > 0 and row["sensitive_gap"] > 0
    ]
    baseline_sensitive_records = [row for row in records if row["baseline"]["sensitive_steps"] > 0]

    if prefer_baseline_sensitive and baseline_sensitive_gap:
        candidate_pool = baseline_sensitive_gap
    elif prefer_baseline_sensitive and baseline_sensitive_records:
        candidate_pool = baseline_sensitive_records
    else:
        candidate_pool = records

    candidate_pool = sorted(candidate_pool, key=_intuition_sort_key, reverse=True)
    selected = []
    seen = set()

    def _add_first_unseen(rows: list[dict], start: int) -> int:
        for idx in range(start, len(rows)):
            row = rows[idx]
            if row["sample_idx"] in seen:
                continue
            selected.append(row)
            seen.add(row["sample_idx"])
            return idx + 1
        return len(rows)

    if balance_labels:
        def _balance_key(row: dict):
            if "balance_concept_value" in row:
                return (row["label_idx"], row["balance_concept_value"])
            return (row["label_idx"],)

        label_order = sorted({_balance_key(row) for row in candidate_pool})
        label_buckets = {
            key: [row for row in candidate_pool if _balance_key(row) == key]
            for key in label_order
        }
        positions = {key: 0 for key in label_order}
        while len(selected) < num_cases and label_order:
            added_this_round = False
            for key in label_order:
                before = len(selected)
                positions[key] = _add_first_unseen(label_buckets[key], positions[key])
                added_this_round = added_this_round or len(selected) > before
                if len(selected) >= num_cases:
                    break
            if not added_this_round:
                break

    for row in candidate_pool:
        if len(selected) >= num_cases:
            break
        if row["sample_idx"] in seen:
            continue
        selected.append(row)
        seen.add(row["sample_idx"])
    return selected
