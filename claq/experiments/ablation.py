"""Factories for the CLAQ ablation variants described in the paper."""

from __future__ import annotations

from copy import deepcopy

import torch

from claq.constraints import ConstraintConfig, EmpiricalConditionalMIScorer
from claq.training.config import ClaqObjectiveConfig


def build_ablation_objectives(
    *,
    query_costs: torch.Tensor,
    query_set_mask: torch.Tensor,
    mi_scorer: EmpiricalConditionalMIScorer,
    lambda_sensitive: float,
    lambda_cost: float,
    lambda_fairness: float,
    lambda_query_set: float = 0.0,
    cost_budget: float | None = None,
    leakage_budget: float | None = None,
    admissible_mask: torch.Tensor | None = None,
    justification_mask: torch.Tensor | None = None,
    proxy_leakage_threshold: float | None = None,
    proxy_label_threshold: float | None = None,
    rollout_steps: int = 10,
    condition_sensitive_on_y: bool = True,
    num_sensitive_classes: int = 2,
    confidence_threshold: float | None = None,
    minimum_rollout_steps: int = 0,
    fairness_tolerance: float | None = None,
    fairness_dual_lr: float = 0.0,
    fairness_lambda_max: float | None = None,
    fairness_min_group_count: int = 2,
    fairness_reduction: str = "max",
) -> dict[str, ClaqObjectiveConfig]:
    """Construct mutually comparable CLAQ ablation configurations.

    All returned variants use the same model class and rollout horizon.  The
    full variant activates every supplied penalty/constraint; earlier variants
    remove one or more components without changing unrelated settings.
    """

    base_constraints = ConstraintConfig(
        query_costs=query_costs,
        admissible_mask=admissible_mask,
        justification_mask=justification_mask,
    )
    cost_constraints = deepcopy(base_constraints)
    cost_constraints.cost_budget = cost_budget
    leak_constraints = deepcopy(base_constraints)
    leak_constraints.leakage_budget = leakage_budget
    leak_cost_constraints = deepcopy(cost_constraints)
    leak_cost_constraints.leakage_budget = leakage_budget
    if (proxy_leakage_threshold is None) != (proxy_label_threshold is None):
        raise ValueError("Both proxy thresholds must be supplied together")
    full_constraints = deepcopy(leak_cost_constraints)
    full_constraints.dynamic_proxy_screen = proxy_leakage_threshold is not None
    full_constraints.proxy_leakage_threshold = proxy_leakage_threshold
    full_constraints.proxy_label_threshold = proxy_label_threshold

    common = dict(
        condition_sensitive_on_y=condition_sensitive_on_y,
        num_sensitive_classes=num_sensitive_classes,
        query_set_mask=query_set_mask,
        mi_scorer=mi_scorer,
        rollout_steps=rollout_steps,
        confidence_threshold=confidence_threshold,
        minimum_rollout_steps=minimum_rollout_steps,
        start_from_sampled_history=False,
    )
    variants = {
        "CLAQ-Base": ClaqObjectiveConfig(constraints=base_constraints, **common),
        "CLAQ-Cost": ClaqObjectiveConfig(
            lambda_cost=lambda_cost,
            constraints=cost_constraints,
            **common,
        ),
        "CLAQ-Leak": ClaqObjectiveConfig(
            lambda_sensitive=lambda_sensitive,
            constraints=leak_constraints,
            **common,
        ),
        "CLAQ-LeakCost": ClaqObjectiveConfig(
            lambda_sensitive=lambda_sensitive,
            lambda_cost=lambda_cost,
            constraints=leak_cost_constraints,
            **common,
        ),
        "CLAQ-Fair": ClaqObjectiveConfig(
            lambda_sensitive=lambda_sensitive,
            lambda_cost=lambda_cost,
            lambda_fairness=lambda_fairness,
            fairness_tolerance=fairness_tolerance,
            fairness_dual_lr=fairness_dual_lr,
            fairness_lambda_max=fairness_lambda_max,
            fairness_min_group_count=fairness_min_group_count,
            fairness_reduction=fairness_reduction,
            constraints=leak_cost_constraints,
            **common,
        ),
        "CLAQ-Full": ClaqObjectiveConfig(
            lambda_sensitive=lambda_sensitive,
            lambda_cost=lambda_cost,
            lambda_fairness=lambda_fairness,
            lambda_query_set=lambda_query_set,
            fairness_tolerance=fairness_tolerance,
            fairness_dual_lr=fairness_dual_lr,
            fairness_lambda_max=fairness_lambda_max,
            fairness_min_group_count=fairness_min_group_count,
            fairness_reduction=fairness_reduction,
            constraints=full_constraints,
            **common,
        ),
    }
    for objective in variants.values():
        objective.validate(query_costs.numel())
    return variants
