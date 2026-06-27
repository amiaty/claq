"""Training configuration for the complete parameterized CLAQ implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from claq.constraints import ConstraintConfig, EmpiricalConditionalMIScorer


@dataclass
class ClaqObjectiveConfig:
    """Loss coefficients and constraint objects used during actor training."""

    lambda_sensitive: float = 0.0
    lambda_cost: float = 0.0
    lambda_fairness: float = 0.0
    lambda_query_set: float = 0.0
    condition_sensitive_on_y: bool = True
    num_sensitive_classes: int = 2
    fairness_min_group_count: int = 2
    fairness_reduction: str = "max"
    fairness_tolerance: float | None = None
    fairness_dual_lr: float = 0.0
    fairness_lambda_max: float | None = None
    query_set_mask: torch.Tensor | None = None
    constraints: ConstraintConfig | None = None
    mi_scorer: EmpiricalConditionalMIScorer | None = None
    rollout_steps: int = 1
    confidence_threshold: float | None = None
    minimum_rollout_steps: int = 0
    start_from_sampled_history: bool = True

    def validate(self, num_queries: int) -> None:
        for name, value in (
            ("lambda_sensitive", self.lambda_sensitive),
            ("lambda_cost", self.lambda_cost),
            ("lambda_fairness", self.lambda_fairness),
            ("lambda_query_set", self.lambda_query_set),
        ):
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.num_sensitive_classes < 2:
            raise ValueError("num_sensitive_classes must be at least 2")
        if self.rollout_steps < 1:
            raise ValueError("rollout_steps must be at least 1")
        if self.minimum_rollout_steps < 0 or self.minimum_rollout_steps > self.rollout_steps:
            raise ValueError("minimum_rollout_steps must lie in [0, rollout_steps]")
        if self.confidence_threshold is not None and not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must lie in (0,1]")
        if self.fairness_tolerance is not None and self.fairness_tolerance < 0:
            raise ValueError("fairness_tolerance must be nonnegative")
        if self.fairness_dual_lr < 0:
            raise ValueError("fairness_dual_lr must be nonnegative")
        if self.fairness_lambda_max is not None and self.fairness_lambda_max < 0:
            raise ValueError("fairness_lambda_max must be nonnegative")
        if self.query_set_mask is not None and self.query_set_mask.numel() != num_queries:
            raise ValueError(f"query_set_mask must contain {num_queries} entries")
        if self.constraints is not None:
            self.constraints.validate(num_queries)
        if (
            self.constraints is not None
            and (self.constraints.dynamic_proxy_screen or self.constraints.leakage_budget is not None)
            and self.mi_scorer is None
        ):
            raise ValueError("dynamic proxy screening and leakage budgets require mi_scorer")
        if self.lambda_cost > 0 and (
            self.constraints is None or self.constraints.query_costs is None
        ):
            raise ValueError("lambda_cost > 0 requires constraints.query_costs")
        if self.lambda_query_set > 0 and self.query_set_mask is None:
            raise ValueError("lambda_query_set > 0 requires query_set_mask")
