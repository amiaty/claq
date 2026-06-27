"""Differentiable losses and empirical metrics used by the CLAQ implementation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def conditional_sensitive_input(
    transcript: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Concatenate a transcript with a one-hot encoding of the task label."""

    one_hot = F.one_hot(labels.to(torch.long), num_classes=num_classes).to(transcript.dtype)
    return torch.cat([transcript, one_hot], dim=1)


def sensitive_prediction_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_sensitive_classes: int,
) -> torch.Tensor:
    """Binary or multiclass sensitive-attribute prediction loss."""

    if num_sensitive_classes == 2:
        return F.binary_cross_entropy_with_logits(logits.view(-1), targets.to(logits.dtype).view(-1))
    return F.cross_entropy(logits, targets.to(torch.long))


def sensitive_prediction_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_sensitive_classes: int,
) -> torch.Tensor:
    if num_sensitive_classes == 2:
        predictions = (torch.sigmoid(logits.view(-1)) >= 0.5).to(targets.dtype)
    else:
        predictions = logits.argmax(dim=1).to(targets.dtype)
    return (predictions == targets.view(-1)).to(torch.float32).mean()


def soft_equalized_odds_penalty(
    class_probabilities: torch.Tensor,
    labels: torch.Tensor,
    sensitive: torch.Tensor,
    *,
    num_label_classes: int,
    num_sensitive_classes: int,
    min_group_count: int = 2,
    reduction: str = "max",
) -> torch.Tensor:
    """Differentiable empirical multiclass equalized-odds penalty.

    For every target class y, predicted class y_hat, and pair of sensitive
    groups s,s', the function compares the group-conditional mean predicted
    probability.  Only cells with at least ``min_group_count`` examples are
    included.  ``reduction='max'`` matches the paper's EO violation; ``mean``
    provides a smoother pairwise surrogate.
    """

    if class_probabilities.ndim != 2 or class_probabilities.size(1) != num_label_classes:
        raise ValueError("class_probabilities must have shape [batch, num_label_classes]")
    differences: list[torch.Tensor] = []
    labels = labels.to(torch.long)
    sensitive = sensitive.to(torch.long)
    for y_value in range(num_label_classes):
        group_means: dict[int, torch.Tensor] = {}
        for s_value in range(num_sensitive_classes):
            idx = (labels == y_value) & (sensitive == s_value)
            if int(idx.sum().item()) >= min_group_count:
                group_means[s_value] = class_probabilities[idx].mean(dim=0)
        group_values = sorted(group_means)
        for left_pos, left_group in enumerate(group_values):
            for right_group in group_values[left_pos + 1 :]:
                differences.append((group_means[left_group] - group_means[right_group]).abs())
    if not differences:
        return class_probabilities.sum() * 0.0
    stacked = torch.cat([value.reshape(-1) for value in differences])
    if reduction == "max":
        return stacked.max()
    if reduction == "mean":
        return stacked.mean()
    raise ValueError("reduction must be 'max' or 'mean'")


@torch.no_grad()
def hard_equalized_odds_details(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    sensitive: torch.Tensor,
    *,
    num_label_classes: int,
    num_sensitive_classes: int,
    min_group_count: int = 1,
) -> dict[str, torch.Tensor | int]:
    """Return hard multiclass equalized-odds diagnostics.

    The violation is the maximum absolute difference in conditional prediction
    rates over target labels, predicted labels, and supported pairs of sensitive
    groups.  The companion counts make sparse-cell behavior explicit.
    """

    if min_group_count < 1:
        raise ValueError("min_group_count must be at least 1")
    predictions = predictions.to(torch.long).view(-1)
    labels = labels.to(torch.long).view(-1)
    sensitive = sensitive.to(torch.long).view(-1)
    if not (predictions.numel() == labels.numel() == sensitive.numel()):
        raise ValueError("predictions, labels, and sensitive must have equal length")
    maximum = torch.zeros((), device=predictions.device, dtype=torch.float32)
    found = False
    valid_group_cells = 0
    valid_group_pairs = 0
    included_counts: list[int] = []
    for y_value in range(num_label_classes):
        rates: dict[int, torch.Tensor] = {}
        for s_value in range(num_sensitive_classes):
            idx = (labels == y_value) & (sensitive == s_value)
            count = int(idx.sum().item())
            if count >= min_group_count:
                rates[s_value] = F.one_hot(
                    predictions[idx], num_classes=num_label_classes
                ).to(torch.float32).mean(dim=0)
                valid_group_cells += 1
                included_counts.append(count)
        groups = sorted(rates)
        for left_pos, left_group in enumerate(groups):
            for right_group in groups[left_pos + 1 :]:
                maximum = torch.maximum(
                    maximum,
                    (rates[left_group] - rates[right_group]).abs().max(),
                )
                valid_group_pairs += 1
                found = True
    violation = maximum if found else torch.tensor(float("nan"), device=predictions.device)
    return {
        "violation": violation,
        "minimum_included_cell_count": min(included_counts) if included_counts else 0,
        "valid_group_cells": valid_group_cells,
        "valid_group_pairs": valid_group_pairs,
    }


@torch.no_grad()
def hard_equalized_odds_violation(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    sensitive: torch.Tensor,
    *,
    num_label_classes: int,
    num_sensitive_classes: int,
    min_group_count: int = 1,
) -> torch.Tensor:
    """Empirical multiclass equalized-odds violation for hard predictions."""

    return hard_equalized_odds_details(
        predictions,
        labels,
        sensitive,
        num_label_classes=num_label_classes,
        num_sensitive_classes=num_sensitive_classes,
        min_group_count=min_group_count,
    )["violation"]


@torch.no_grad()
def macro_f1_score(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    predictions = predictions.to(torch.long).view(-1)
    labels = labels.to(torch.long).view(-1)
    scores = []
    for class_index in range(num_classes):
        tp = ((predictions == class_index) & (labels == class_index)).sum().to(torch.float32)
        fp = ((predictions == class_index) & (labels != class_index)).sum().to(torch.float32)
        fn = ((predictions != class_index) & (labels == class_index)).sum().to(torch.float32)
        denom = 2 * tp + fp + fn
        scores.append(torch.where(denom > 0, 2 * tp / denom, torch.zeros_like(denom)))
    return torch.stack(scores).mean()
