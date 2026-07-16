"""Shared fixed-horizon evaluation for the joint-control experiments."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


@torch.inference_mode()
def fixed_horizon_rollout(
    *,
    actor,
    classifier,
    answers: torch.Tensor,
    labels: torch.Tensor,
    sensitive_targets: torch.Tensor,
    cost_vector: torch.Tensor,
    sensitive_mask: torch.Tensor,
    horizon: int,
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    """Roll out from an empty knowledge state for exactly ``horizon`` queries."""
    if answers.ndim != 2:
        raise ValueError("answers must have shape [samples, queries]")
    sample_count, query_count = answers.shape
    if not 1 <= horizon <= query_count:
        raise ValueError("horizon must lie in [1, number of queries]")
    if labels.ndim != 1 or sensitive_targets.ndim != 1:
        raise ValueError("labels and sensitive_targets must be one-dimensional")
    if len(labels) != sample_count or len(sensitive_targets) != sample_count:
        raise ValueError("answers, labels, and sensitive_targets must align")
    if tuple(cost_vector.shape) != (query_count,):
        raise ValueError("cost_vector must contain one value per query")
    if tuple(sensitive_mask.shape) != (query_count,):
        raise ValueError("sensitive_mask must contain one value per query")

    actor.eval()
    classifier.eval()
    costs = cost_vector.to(device=device, dtype=torch.float32)
    designated = sensitive_mask.to(device=device, dtype=torch.float32)
    states = []
    predictions = []
    realized_costs = []
    sensitive_counts = []

    for start in range(0, sample_count, batch_size):
        batch_answers = answers[start : start + batch_size].to(
            device=device, dtype=torch.float32
        )
        mask = torch.zeros_like(batch_answers)
        knowledge_state = torch.zeros_like(batch_answers)
        cumulative_cost = torch.zeros(len(batch_answers), device=device)
        cumulative_sensitive = torch.zeros(len(batch_answers), device=device)

        for _ in range(horizon):
            query_distribution = actor(knowledge_state, mask)
            cumulative_cost += (query_distribution * costs).sum(dim=1)
            cumulative_sensitive += (query_distribution * designated).sum(dim=1)
            knowledge_state = knowledge_state + query_distribution * batch_answers
            mask = torch.clamp(mask + query_distribution, 0.0, 1.0)

        states.append(knowledge_state.cpu())
        predictions.append(classifier(knowledge_state).argmax(dim=1).cpu())
        realized_costs.append(cumulative_cost.cpu())
        sensitive_counts.append(cumulative_sensitive.cpu())

    return {
        "knowledge_states": torch.cat(states).numpy(),
        "labels": labels.cpu().numpy(),
        "sensitive_targets": sensitive_targets.cpu().numpy(),
        "predictions": torch.cat(predictions).numpy(),
        "realized_costs": torch.cat(realized_costs).numpy(),
        "sensitive_query_counts": torch.cat(sensitive_counts).numpy(),
    }


def summarize_fixed_horizon(
    rollout: dict[str, np.ndarray],
    *,
    horizon: int,
    include_macro_f1: bool,
) -> dict[str, float]:
    """Summarize one seed's held-out fixed-horizon rollout."""
    labels = rollout["labels"]
    predictions = rollout["predictions"]
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "mean_cumulative_cost": float(np.mean(rollout["realized_costs"])),
        "sensitive_query_rate": float(
            np.mean(rollout["sensitive_query_counts"]) / horizon
        ),
    }
    if include_macro_f1:
        result["macro_f1"] = float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        )
    return result
