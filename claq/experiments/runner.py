"""Reproducible single- and multi-seed runners for CLAQ ablations."""

from __future__ import annotations

from copy import copy
from pathlib import Path

import torch

from claq.core.checkpoints import save_bundle_checkpoint
from claq.training import (
    ClaqObjectiveConfig,
    HistorySamplingConfig,
    build_claq_models,
    evaluate_claq,
    fit_claq,
    seed_everything,
)


def _clone_objective(objective: ClaqObjectiveConfig) -> ClaqObjectiveConfig:
    """Clone mutable scalar state without duplicating the MI reference sample."""

    cloned = copy(objective)
    cloned.constraints = copy(objective.constraints) if objective.constraints is not None else None
    return cloned


def train_claq_variant(
    *,
    variant_name: str,
    seed: int,
    objective_config: ClaqObjectiveConfig,
    train_loader,
    validation_loader,
    test_loader,
    model_clip,
    dictionary: torch.Tensor,
    answering_model,
    sensitive_indices: torch.Tensor,
    history_config: HistorySamplingConfig,
    clip_device: torch.device,
    train_device: torch.device,
    threshold_for_binarization: float,
    sensitive_tau: float,
    sensitive_topk: int,
    num_classes: int,
    num_epochs: int,
    learning_rate: float,
    actor_eps: float = 1.0,
    actor_eps_end: float | None = None,
    actor_eps_anneal_epochs: int | None = None,
    sensitive_target_mode: str = "hard",
    sensitive_target_indices: torch.Tensor | None = None,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
    max_test_batches: int | None = None,
    selection_metric: str = "acc",
    selection_mode: str = "max",
    checkpoint_path: str | Path | None = None,
    deterministic: bool = False,
) -> dict:
    """Train one CLAQ configuration and evaluate the selected checkpoint once."""

    seed_everything(seed, deterministic=deterministic)
    objective = _clone_objective(objective_config)
    actor, classifier, sensitive_head = build_claq_models(
        max_queries=int(dictionary.size(1)),
        num_classes=num_classes,
        device=train_device,
        actor_eps=actor_eps,
        condition_sensitive_on_y=objective.condition_sensitive_on_y,
        num_sensitive_classes=objective.num_sensitive_classes,
    )
    optimizer = torch.optim.Adam(
        list(actor.parameters())
        + list(classifier.parameters())
        + list(sensitive_head.parameters()),
        lr=learning_rate,
    )
    history, best = fit_claq(
        actor=actor,
        classifier=classifier,
        s_head=sensitive_head,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=validation_loader,
        model_clip=model_clip,
        dictionary=dictionary,
        answering_model=answering_model,
        sens_idx=sensitive_indices,
        history_config=history_config,
        clip_device=clip_device,
        train_device=train_device,
        threshold_for_binarization=threshold_for_binarization,
        sensitive_tau=sensitive_tau,
        sensitive_topk=sensitive_topk,
        num_epochs=num_epochs,
        max_train_batches=max_train_batches,
        max_test_batches=max_validation_batches,
        actor_eps_end=actor_eps_end,
        actor_eps_anneal_epochs=actor_eps_anneal_epochs,
        sensitive_target_mode=sensitive_target_mode,
        sensitive_target_indices=sensitive_target_indices,
        objective_config=objective,
        selection_metric=selection_metric,
        selection_mode=selection_mode,
    )
    actor.load_state_dict(best["actor_state_dict"])
    classifier.load_state_dict(best["classifier_state_dict"])
    sensitive_head.load_state_dict(best["s_head_state_dict"])
    test_metrics = evaluate_claq(
        loader=test_loader,
        actor=actor,
        classifier=classifier,
        s_head=sensitive_head,
        model_clip=model_clip,
        dictionary=dictionary,
        answering_model=answering_model,
        sens_idx=sensitive_indices,
        history_config=history_config,
        clip_device=clip_device,
        train_device=train_device,
        threshold_for_binarization=threshold_for_binarization,
        sensitive_tau=sensitive_tau,
        sensitive_topk=sensitive_topk,
        sensitive_target_mode=sensitive_target_mode,
        sensitive_target_indices=sensitive_target_indices,
        objective_config=objective,
        max_batches=max_test_batches,
    )
    if checkpoint_path is not None:
        save_bundle_checkpoint(
            checkpoint_path=checkpoint_path,
            actor=actor,
            classifier=classifier,
            s_head=sensitive_head,
            optimizer=None,
            metadata={
                "variant_name": variant_name,
                "seed": int(seed),
                "best_epoch": int(best["epoch"]),
                "validation_score": float(best["validation_score"]),
                "test_metrics": test_metrics,
                "condition_sensitive_on_y": bool(objective.condition_sensitive_on_y),
                "num_sensitive_classes": int(objective.num_sensitive_classes),
            },
        )
    return {
        "variant_name": variant_name,
        "seed": int(seed),
        "history": history,
        "best": best,
        "test_metrics": test_metrics,
        "actor": actor,
        "classifier": classifier,
        "sensitive_head": sensitive_head,
        "objective_config": objective,
    }


def run_multi_seed_ablation(
    *,
    objectives: dict[str, ClaqObjectiveConfig],
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    checkpoints_root: str | Path | None = None,
    **train_kwargs,
) -> list[dict]:
    """Train every variant on every seed using an identical experimental setup."""

    results = []
    root = Path(checkpoints_root) if checkpoints_root is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    for variant_name, objective in objectives.items():
        for seed in seeds:
            checkpoint_path = None
            if root is not None:
                checkpoint_path = root / f"{variant_name}_seed{seed}.pt"
            results.append(
                train_claq_variant(
                    variant_name=variant_name,
                    seed=seed,
                    objective_config=objective,
                    checkpoint_path=checkpoint_path,
                    **train_kwargs,
                )
            )
    return results
