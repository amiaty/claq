"""Constraint and empirical information-score utilities for CLAQ.

The population CLAQ formulation uses transcript-conditional mutual-information
quantities.  This module provides a finite-sample plug-in implementation for
binary concept responses and discrete target/sensitive variables.  It is meant
for feasibility screening and diagnostics; the neural actor is still trained
with differentiable task/adversarial surrogates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch


class ConditionalMIScores(NamedTuple):
    """Candidate-wise conditional information scores for one or more histories."""

    label_information: torch.Tensor
    sensitive_leakage: torch.Tensor
    support: torch.Tensor


def _as_binary_long(values: torch.Tensor) -> torch.Tensor:
    """Validate hard binary responses and map them to ``{0,1}``.

    The scorer intentionally rejects probabilities and logits.  Information
    scores must be computed from the same finite-alphabet operational responses
    that are inserted into the transcript.
    """

    if values.ndim != 2:
        raise ValueError(f"Expected a rank-2 response tensor, received shape {tuple(values.shape)}")
    values = values.detach()
    if values.dtype == torch.bool:
        return values.to(torch.long)
    is_zero_one = bool(((values == 0) | (values == 1)).all().item())
    is_minus_plus = bool(((values == -1) | (values == 1)).all().item())
    if not (is_zero_one or is_minus_plus):
        raise ValueError("responses must be hard binary values encoded as {0,1} or {-1,+1}")
    return (values > 0).to(torch.long)


def _mutual_information_from_counts(counts: torch.Tensor, alpha: float) -> torch.Tensor:
    """Return plug-in I(A;B) from a nonnegative |A| x |B| count table."""

    counts = counts.to(torch.float64)
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    if alpha > 0:
        counts = counts + alpha
    total = counts.sum()
    if total <= 0:
        return torch.zeros((), dtype=torch.float64)
    p_ab = counts / total
    p_a = p_ab.sum(dim=1, keepdim=True)
    p_b = p_ab.sum(dim=0, keepdim=True)
    denom = p_a * p_b
    valid = p_ab > 0
    return torch.where(valid, p_ab * torch.log2(p_ab / denom.clamp_min(torch.finfo(torch.float64).tiny)), 0.0).sum()


def _conditional_mutual_information_s_r_given_y(
    s: torch.Tensor,
    r: torch.Tensor,
    y: torch.Tensor,
    num_sensitive_classes: int,
    num_label_classes: int,
    alpha: float,
) -> torch.Tensor:
    """Return the plug-in estimate of I(S;R | Y) for binary R."""

    total = int(y.numel())
    if total == 0:
        return torch.zeros((), dtype=torch.float64)
    result = torch.zeros((), dtype=torch.float64)
    for y_value in range(num_label_classes):
        idx = y == y_value
        n_y = int(idx.sum().item())
        if n_y == 0:
            continue
        counts = torch.zeros((num_sensitive_classes, 2), dtype=torch.float64)
        s_y = s[idx]
        r_y = r[idx]
        flat = s_y * 2 + r_y
        bincount = torch.bincount(flat, minlength=num_sensitive_classes * 2).to(torch.float64)
        counts.copy_(bincount.view(num_sensitive_classes, 2))
        result = result + (n_y / total) * _mutual_information_from_counts(counts, alpha)
    return result


def _mutual_information_y_r(
    y: torch.Tensor,
    r: torch.Tensor,
    num_label_classes: int,
    alpha: float,
) -> torch.Tensor:
    counts = torch.zeros((num_label_classes, 2), dtype=torch.float64)
    flat = y * 2 + r
    bincount = torch.bincount(flat, minlength=num_label_classes * 2).to(torch.float64)
    counts.copy_(bincount.view(num_label_classes, 2))
    return _mutual_information_from_counts(counts, alpha)


class EmpiricalConditionalMIScorer:
    """Finite-sample conditional-MI scorer for discrete CLAQ histories.

    For a realized history ``h`` encoded by ``mask`` and ``masked_answers``, the
    scorer restricts a reference sample to examples whose responses agree with
    every observed query response.  For each unobserved query q it then computes

        I_hat(Y; R_q | H=h)
        I_hat(S; R_q | Y, H=h).

    Histories with fewer than ``min_support`` matching reference examples fall
    back to the unconditional scores.  Results are memoized by the exact binary
    history key.  This procedure is statistically transparent but can be
    expensive for large reference sets; it is intended for constraint masks,
    evaluation, and moderate-size experiments rather than as a differentiable
    training loss.
    """

    def __init__(
        self,
        answers: torch.Tensor,
        labels: torch.Tensor,
        sensitive: torch.Tensor,
        *,
        num_label_classes: int | None = None,
        num_sensitive_classes: int | None = None,
        alpha: float = 0.5,
        min_support: int = 32,
        cache_size: int = 100_000,
    ) -> None:
        if labels.ndim != 1 or sensitive.ndim != 1:
            raise ValueError("labels and sensitive must be rank-1 tensors")
        if answers.size(0) != labels.numel() or labels.numel() != sensitive.numel():
            raise ValueError("answers, labels, and sensitive must contain the same number of examples")
        if labels.numel() == 0:
            raise ValueError("the reference sample must not be empty")
        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        if min_support < 1:
            raise ValueError("min_support must be at least 1")
        if cache_size < 1:
            raise ValueError("cache_size must be at least 1")
        self.answers = _as_binary_long(answers.detach().cpu())
        self.labels = labels.detach().cpu().to(torch.long)
        self.sensitive = sensitive.detach().cpu().to(torch.long)
        if (self.labels < 0).any() or (self.sensitive < 0).any():
            raise ValueError("labels and sensitive values must be nonnegative integer class indices")
        self.num_queries = int(self.answers.size(1))
        inferred_label_classes = int(self.labels.max().item()) + 1
        inferred_sensitive_classes = int(self.sensitive.max().item()) + 1
        self.num_label_classes = int(num_label_classes or inferred_label_classes)
        self.num_sensitive_classes = int(num_sensitive_classes or inferred_sensitive_classes)
        if self.num_label_classes < inferred_label_classes:
            raise ValueError("num_label_classes is smaller than an observed label index")
        if self.num_sensitive_classes < inferred_sensitive_classes:
            raise ValueError("num_sensitive_classes is smaller than an observed sensitive index")
        self.alpha = float(alpha)
        self.min_support = int(min_support)
        self.cache_size = int(cache_size)
        self._cache: dict[tuple[tuple[int, int], ...], tuple[torch.Tensor, torch.Tensor, int]] = {}
        self._global = self._compute_scores(torch.ones(self.answers.size(0), dtype=torch.bool))


    @property
    def global_scores(self) -> ConditionalMIScores:
        label_scores, leakage_scores, support = self._global
        return ConditionalMIScores(
            label_information=label_scores.to(torch.float32).clone(),
            sensitive_leakage=leakage_scores.to(torch.float32).clone(),
            support=torch.full((self.num_queries,), support, dtype=torch.long),
        )

    def _history_key(self, mask: torch.Tensor, masked_answers: torch.Tensor) -> tuple[tuple[int, int], ...]:
        mask_cpu = mask.detach().cpu().view(-1) > 0.5
        values_cpu = (masked_answers.detach().cpu().view(-1) > 0).to(torch.long)
        indices = mask_cpu.nonzero(as_tuple=False).flatten().tolist()
        return tuple((int(index), int(values_cpu[index].item())) for index in indices)

    def _matching_rows(self, key: tuple[tuple[int, int], ...]) -> torch.Tensor:
        if not key:
            return torch.ones(self.answers.size(0), dtype=torch.bool)
        result = torch.ones(self.answers.size(0), dtype=torch.bool)
        for query_index, response_value in key:
            result &= self.answers[:, query_index] == response_value
        return result

    def _compute_scores(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        support = int(rows.sum().item())
        label_scores = torch.zeros(self.num_queries, dtype=torch.float64)
        leakage_scores = torch.zeros(self.num_queries, dtype=torch.float64)
        if support == 0:
            return label_scores, leakage_scores, support
        y = self.labels[rows]
        s = self.sensitive[rows]
        candidate_answers = self.answers[rows]
        for q in range(self.num_queries):
            r = candidate_answers[:, q]
            label_scores[q] = _mutual_information_y_r(
                y=y,
                r=r,
                num_label_classes=self.num_label_classes,
                alpha=self.alpha,
            )
            leakage_scores[q] = _conditional_mutual_information_s_r_given_y(
                s=s,
                r=r,
                y=y,
                num_sensitive_classes=self.num_sensitive_classes,
                num_label_classes=self.num_label_classes,
                alpha=self.alpha,
            )
        return label_scores, leakage_scores, support

    def score_history(
        self,
        mask: torch.Tensor,
        masked_answers: torch.Tensor,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> ConditionalMIScores:
        key = self._history_key(mask, masked_answers)
        cached = self._cache.get(key)
        if cached is None:
            rows = self._matching_rows(key)
            scores = self._compute_scores(rows)
            if len(self._cache) >= self.cache_size:
                # Deterministic FIFO behavior is unnecessary here; clearing keeps
                # memory bounded without adding another dependency.
                self._cache.clear()
            self._cache[key] = scores
            cached = scores
        label_scores, leakage_scores, support = cached
        if support < self.min_support:
            label_scores, leakage_scores, _ = self._global
        target_device = device or mask.device
        support_tensor = torch.full((self.num_queries,), support, dtype=torch.long, device=target_device)
        return ConditionalMIScores(
            label_information=label_scores.to(device=target_device, dtype=dtype),
            sensitive_leakage=leakage_scores.to(device=target_device, dtype=dtype),
            support=support_tensor,
        )

    def score_batch(
        self,
        masks: torch.Tensor,
        masked_answers: torch.Tensor,
    ) -> ConditionalMIScores:
        if masks.shape != masked_answers.shape:
            raise ValueError("masks and masked_answers must have the same shape")
        label_rows = []
        leakage_rows = []
        support_rows = []
        for mask, history in zip(masks, masked_answers, strict=True):
            row = self.score_history(mask, history, device=masks.device, dtype=masked_answers.dtype)
            label_rows.append(row.label_information)
            leakage_rows.append(row.sensitive_leakage)
            support_rows.append(row.support)
        return ConditionalMIScores(
            label_information=torch.stack(label_rows, dim=0),
            sensitive_leakage=torch.stack(leakage_rows, dim=0),
            support=torch.stack(support_rows, dim=0),
        )


@dataclass
class ConstraintConfig:
    """Executable feasibility and penalty configuration for a CLAQ policy."""

    query_costs: torch.Tensor | None = None
    cost_budget: float | None = None
    leakage_budget: float | None = None
    admissible_mask: torch.Tensor | None = None
    justification_mask: torch.Tensor | None = None
    proxy_leakage_threshold: float | None = None
    proxy_label_threshold: float | None = None
    dynamic_proxy_screen: bool = False

    def validate(self, num_queries: int) -> None:
        for name, tensor in (
            ("query_costs", self.query_costs),
            ("admissible_mask", self.admissible_mask),
            ("justification_mask", self.justification_mask),
        ):
            if tensor is not None and (tensor.ndim != 1 or tensor.numel() != num_queries):
                raise ValueError(f"{name} must be a length-{num_queries} vector")
        for name, tensor in (
            ("admissible_mask", self.admissible_mask),
            ("justification_mask", self.justification_mask),
        ):
            if tensor is not None and tensor.dtype != torch.bool:
                raise ValueError(f"{name} must have boolean dtype")
        if self.query_costs is not None:
            if not torch.isfinite(self.query_costs).all():
                raise ValueError("query costs must be finite")
            if (self.query_costs < 0).any():
                raise ValueError("query costs must be nonnegative")
        if self.cost_budget is not None and (not torch.isfinite(torch.tensor(self.cost_budget)) or self.cost_budget < 0):
            raise ValueError("cost_budget must be finite and nonnegative, or None")
        if self.leakage_budget is not None and (not torch.isfinite(torch.tensor(self.leakage_budget)) or self.leakage_budget < 0):
            raise ValueError("leakage_budget must be finite and nonnegative, or None")
        if self.proxy_leakage_threshold is not None and (
            not torch.isfinite(torch.tensor(self.proxy_leakage_threshold))
            or self.proxy_leakage_threshold < 0
        ):
            raise ValueError("proxy_leakage_threshold must be finite and nonnegative")
        if self.proxy_label_threshold is not None and (
            not torch.isfinite(torch.tensor(self.proxy_label_threshold))
            or self.proxy_label_threshold < 0
        ):
            raise ValueError("proxy_label_threshold must be finite and nonnegative")
        if self.dynamic_proxy_screen:
            if self.proxy_leakage_threshold is None or self.proxy_label_threshold is None:
                raise ValueError(
                    "dynamic proxy screening requires both proxy_leakage_threshold and proxy_label_threshold"
                )


def build_unavailable_action_mask(
    queried_mask: torch.Tensor,
    *,
    config: ConstraintConfig | None = None,
    scores: ConditionalMIScores | None = None,
    cumulative_cost: torch.Tensor | None = None,
    cumulative_leakage: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a Boolean mask whose true entries are infeasible actions."""

    if queried_mask.ndim != 2:
        raise ValueError("queried_mask must have shape [batch, num_queries]")
    unavailable = queried_mask > 0.5
    if config is None:
        return unavailable

    batch_size, num_queries = queried_mask.shape
    config.validate(num_queries)
    device = queried_mask.device

    if config.admissible_mask is not None:
        admissible = config.admissible_mask.to(device=device, dtype=torch.bool).view(1, -1)
        unavailable = unavailable | (~admissible)

    if config.cost_budget is not None:
        if config.query_costs is None:
            raise ValueError("cost_budget requires query_costs")
        costs = config.query_costs.to(device=device, dtype=torch.float32).view(1, -1)
        if cumulative_cost is None:
            cumulative_cost = (queried_mask.to(torch.float32) * costs).sum(dim=1)
        candidate_total = cumulative_cost.view(batch_size, 1) + costs
        unavailable = unavailable | (candidate_total > float(config.cost_budget) + 1e-12)

    if config.leakage_budget is not None:
        if scores is None:
            raise ValueError("leakage_budget requires candidate conditional-MI scores")
        if cumulative_leakage is None:
            cumulative_leakage = torch.zeros(batch_size, device=device, dtype=scores.sensitive_leakage.dtype)
        candidate_total = cumulative_leakage.view(batch_size, 1) + scores.sensitive_leakage
        unavailable = unavailable | (candidate_total > float(config.leakage_budget) + 1e-12)

    if config.dynamic_proxy_screen:
        if scores is None:
            raise ValueError("dynamic proxy screening requires candidate conditional-MI scores")
        proxy = (
            (scores.sensitive_leakage > float(config.proxy_leakage_threshold))
            & (scores.label_information < float(config.proxy_label_threshold))
        )
        if config.justification_mask is not None:
            justified = config.justification_mask.to(device=device, dtype=torch.bool).view(1, -1)
            proxy = proxy & (~justified)
        unavailable = unavailable | proxy

    return unavailable


def build_static_proxy_admissible_mask(
    scorer: EmpiricalConditionalMIScorer,
    *,
    leakage_threshold: float,
    label_information_threshold: float,
    justification_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the static admissibility mask induced by the paper's proxy rule."""

    scores = scorer.global_scores
    proxy = (
        (scores.sensitive_leakage > float(leakage_threshold))
        & (scores.label_information < float(label_information_threshold))
    )
    if justification_mask is not None:
        if justification_mask.numel() != scorer.num_queries:
            raise ValueError("justification_mask has the wrong length")
        proxy = proxy & (~justification_mask.to(torch.bool).cpu())
    return ~proxy
