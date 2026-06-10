"""Configurable history sampling for VIP training and analysis."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class HistorySamplingConfig:
    min_history: int = 1
    max_history: int = 2
    non_sensitive_only: bool = True


def sample_history_mask(
    answers: torch.Tensor,
    config: HistorySamplingConfig,
    sensitive_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_queries = answers.shape
    mask = torch.zeros_like(answers)

    if config.non_sensitive_only:
        if sensitive_indices is None:
            raise ValueError("sensitive_indices are required when non_sensitive_only=True")
        sensitive_set = set(sensitive_indices.tolist())
        pool = [idx for idx in range(num_queries) if idx not in sensitive_set]
    else:
        pool = list(range(num_queries))

    if not pool:
        return mask, answers * mask

    max_history = min(config.max_history, len(pool), num_queries - 1)
    min_history = min(config.min_history, max_history)

    for row in range(batch_size):
        history_size = int(torch.randint(low=min_history, high=max_history + 1, size=(1,), device=answers.device).item())
        if history_size == 0:
            continue
        chosen = torch.tensor(pool, device=answers.device)[torch.randperm(len(pool), device=answers.device)[:history_size]]
        mask[row, chosen] = 1.0

    return mask, answers * mask
