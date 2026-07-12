"""Per-query cost annotation (framework Step 4).

Implements the two cost specifications from the CLAQ framework:

- Uniform query-count cost: every query costs 1. Cumulative cost equals the
  number of executed queries, and the training cost term does not change the
  actor's ordering (see :func:`expected_query_cost`).
- Synthetic response-uncertainty cost: learned concept queries are priced at
  ``1 + D(q)``, where ``D(q)`` is the normalized predictive entropy of the soft
  response model averaged over a calibration set. For binary responses the
  normalizer ``log2(|R|)`` equals 1, so ``D(q)`` lies in ``[0, 1]``.

The cost function is fixed before policy training and does not depend on test
data. Protected-attribute leakage is represented separately and is not part of
the cost.
"""

from __future__ import annotations

import torch


def normalized_binary_entropy(prob: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Base-2 Bernoulli entropy of ``P(response=1)=prob``.

    For a binary response the entropy is already normalized to ``[0, 1]`` because
    the normalizer ``log2(|R|) = log2(2) = 1``.
    """
    p = prob.clamp(eps, 1.0 - eps)
    return -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p))


def build_uniform_cost(num_queries: int, device: torch.device | None = None) -> torch.Tensor:
    """Uniform cost ``C_unit(q) = 1`` for every query."""
    return torch.ones(num_queries, device=device)


def build_uncertainty_cost(
    soft_responses: torch.Tensor,
    learned_mask: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Synthetic response-uncertainty cost ``C_unc(q) = 1 + D(q)``.

    Args:
        soft_responses: ``[N_cal, num_queries]`` calibration probabilities
            ``P(response=1 | x, c)`` produced by the soft response model.
        learned_mask: optional ``[num_queries]`` mask marking learned concept
            queries. Queries outside the mask keep unit cost. When omitted, every
            query is treated as a learned concept query.
        device: optional device for the returned cost vector.

    Returns:
        A ``[num_queries]`` cost vector in ``[1, 2]``.
    """
    if soft_responses.dim() != 2:
        raise ValueError("soft_responses must have shape [N_cal, num_queries]")
    d = normalized_binary_entropy(soft_responses).mean(dim=0)
    if learned_mask is not None:
        d = d * learned_mask.to(d.device).float()
    cost = 1.0 + d
    if device is not None:
        cost = cost.to(device)
    return cost


def expected_query_cost(query_distribution: torch.Tensor, cost_vector: torch.Tensor) -> torch.Tensor:
    """Mean expected per-step cost ``E_q[C(q)]`` used as the training cost penalty.

    With the actor's straight-through one-hot selection this equals the cost of
    the selected query. Under uniform cost the value is constant, so its gradient
    is exactly zero and the query ordering is unchanged. Centering the costs
    prevents numerical gradients from a theoretically constant softmax sum.
    """
    costs = cost_vector.to(query_distribution.device)
    base_cost = costs[0]
    relative_cost = costs - base_cost
    return base_cost + (query_distribution * relative_cost).sum(dim=1).mean()
