"""CLAQ training for datasets with precomputed finite-alphabet query responses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from claq.objectives import (
    hard_equalized_odds_violation,
    macro_f1_score,
    sensitive_prediction_accuracy,
    sensitive_prediction_loss,
    soft_equalized_odds_penalty,
)
from claq.training.claq import _select_and_update, _sensitive_head_input
from claq.training.config import ClaqObjectiveConfig
from claq.training.history_sampling import HistorySamplingConfig, sample_history_mask


def _cpu_state_dict(module) -> dict[str, torch.Tensor]:
    """Clone a module state to CPU for a device-independent checkpoint."""

    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def run_claq_tensor_epoch(
    *,
    loader,
    actor,
    classifier,
    s_head,
    optimizer,
    objective_config: ClaqObjectiveConfig,
    history_config: HistorySamplingConfig,
    device: torch.device,
    train: bool,
    max_batches: int | None = None,
) -> dict:
    """Train/evaluate CLAQ when each batch is ``(responses, Y, S)``.

    ``responses`` must have shape ``[batch, num_queries]`` and contain binary
    values represented as either ``{-1,+1}`` or ``{0,1}``.  Values in ``{0,1}``
    are converted to ``{-1,+1}`` so that zero remains reserved for unqueried
    coordinates in the transcript state.
    """

    objective = objective_config
    objective.validate(actor.output_dim)
    expected_sensitive_input = actor.output_dim + classifier.output_dim if objective.condition_sensitive_on_y else actor.output_dim
    if int(s_head.query_size) != int(expected_sensitive_input):
        raise ValueError("The sensitive head input dimension is inconsistent with the objective")
    if objective.start_from_sampled_history and objective.constraints is not None and objective.constraints.leakage_budget is not None:
        raise ValueError("Pathwise leakage budgets require start_from_sampled_history=False")

    actor.train(train)
    classifier.train(train)
    s_head.train(train)
    if train and optimizer is None:
        raise ValueError("optimizer is required when train=True")

    sums = {name: 0.0 for name in [
        "loss", "task", "sens", "sens_acc", "cost", "fairness", "qset",
        "query_entropy", "set_query_rate", "realized_leakage", "mean_queries", "actor_grad",
    ]}
    n_batches = 0
    total = 0
    correct = 0
    predictions_all = []
    labels_all = []
    sensitive_all = []

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        if len(batch) != 3:
            raise ValueError("Tensor CLAQ batches must contain (responses, labels, sensitive_targets)")
        answers, labels, sensitive = batch
        answers = answers.to(device).to(torch.float32)
        if bool(((answers == 0) | (answers == 1)).all().item()):
            answers = answers * 2.0 - 1.0
        if not bool(((answers == -1) | (answers == 1)).all().item()):
            raise ValueError("responses must be binary and encoded as {-1,+1} or {0,1}")
        labels = labels.to(device).to(torch.long)
        sensitive = sensitive.to(device)
        if objective.num_sensitive_classes == 2:
            sensitive_target = sensitive.to(torch.float32).view(-1)
            sensitive_hard = (sensitive_target >= 0.5).to(torch.long)
        else:
            sensitive_target = sensitive.to(torch.long).view(-1)
            sensitive_hard = sensitive_target

        with torch.set_grad_enabled(train):
            if objective.start_from_sampled_history:
                queried_mask, state = sample_history_mask(
                    answers=answers,
                    config=history_config,
                    sensitive_indices=(objective.query_set_mask > 0.5).nonzero(as_tuple=False).flatten()
                    if objective.query_set_mask is not None
                    else torch.empty(0, dtype=torch.long, device=device),
                )
            else:
                queried_mask = torch.zeros_like(answers)
                state = torch.zeros_like(answers)

            batch_size = answers.size(0)
            cumulative_cost = torch.zeros(batch_size, device=device)
            if objective.constraints is not None and objective.constraints.query_costs is not None:
                costs = objective.constraints.query_costs.to(device=device, dtype=answers.dtype)
                cumulative_cost = (queried_mask * costs.view(1, -1)).sum(dim=1)
            cumulative_leakage = torch.zeros(batch_size, device=device)
            total_cost = torch.zeros(batch_size, device=device)
            total_leakage = torch.zeros(batch_size, device=device)
            total_qset = torch.zeros(batch_size, device=device)
            query_count = torch.zeros(batch_size, device=device)
            entropy_terms = []
            set_rate_terms = []
            active_rows = torch.ones(batch_size, device=device, dtype=torch.bool)

            for rollout_step in range(objective.rollout_steps):
                if objective.confidence_threshold is not None and rollout_step >= objective.minimum_rollout_steps:
                    with torch.no_grad():
                        confidence = F.softmax(classifier(state), dim=1).max(dim=1).values
                    active_rows &= confidence < float(objective.confidence_threshold)
                if not bool(active_rows.any().item()):
                    break
                (
                    state,
                    queried_mask,
                    query_distribution,
                    selected_cost,
                    selected_leakage,
                    selected_qset,
                    selected_active,
                    soft_distribution,
                ) = _select_and_update(
                    actor=actor,
                    state=state,
                    answers=answers,
                    queried_mask=queried_mask,
                    objective=objective,
                    cumulative_cost=cumulative_cost,
                    cumulative_leakage=cumulative_leakage,
                    active_rows=active_rows,
                )
                active_rows &= selected_active
                if objective.constraints is not None and objective.constraints.query_costs is not None:
                    costs = objective.constraints.query_costs.to(device=device, dtype=answers.dtype)
                    cumulative_cost = (queried_mask.detach() * costs.view(1, -1)).sum(dim=1)
                cumulative_leakage += selected_leakage.detach()
                total_cost += selected_cost
                total_leakage += selected_leakage.detach()
                total_qset += selected_qset
                query_count += selected_active.to(torch.float32)
                safe = soft_distribution.clamp_min(1e-8)
                entropy_terms.append((-(safe * torch.log(safe)).sum(dim=1) * selected_active).mean())
                if objective.query_set_mask is not None:
                    qset = objective.query_set_mask.to(device=device, dtype=answers.dtype)
                    set_rate_terms.append(((query_distribution * qset.view(1, -1)).sum(dim=1) * selected_active).mean())

            logits = classifier(state)
            loss_task = F.cross_entropy(logits, labels)
            sensitive_input = _sensitive_head_input(
                state,
                labels,
                condition_on_y=objective.condition_sensitive_on_y,
                num_label_classes=classifier.output_dim,
                lambda_sensitive=objective.lambda_sensitive,
            )
            sensitive_logits = s_head(sensitive_input)
            loss_sensitive = sensitive_prediction_loss(
                sensitive_logits, sensitive_target, objective.num_sensitive_classes
            )
            sensitive_accuracy = sensitive_prediction_accuracy(
                sensitive_logits.detach(), sensitive_target.detach(), objective.num_sensitive_classes
            )
            loss_cost = total_cost.mean()
            loss_qset = total_qset.mean()
            if objective.lambda_fairness > 0 or objective.fairness_tolerance is not None:
                fairness = soft_equalized_odds_penalty(
                    F.softmax(logits, dim=1),
                    labels,
                    sensitive_hard,
                    num_label_classes=classifier.output_dim,
                    num_sensitive_classes=objective.num_sensitive_classes,
                    min_group_count=objective.fairness_min_group_count,
                    reduction=objective.fairness_reduction,
                )
            else:
                fairness = logits.sum() * 0.0
            loss = (
                loss_task
                + loss_sensitive
                + objective.lambda_cost * loss_cost
                + objective.lambda_query_set * loss_qset
                + objective.lambda_fairness * fairness
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                actor_grad = sum(
                    float(parameter.grad.detach().norm().item())
                    for parameter in actor.parameters()
                    if parameter.grad is not None
                )
                optimizer.step()
                sums["actor_grad"] += actor_grad
                if objective.fairness_tolerance is not None and objective.fairness_dual_lr > 0:
                    value = max(
                        0.0,
                        objective.lambda_fairness
                        + objective.fairness_dual_lr
                        * (float(fairness.detach().item()) - objective.fairness_tolerance),
                    )
                    if objective.fairness_lambda_max is not None:
                        value = min(value, objective.fairness_lambda_max)
                    objective.lambda_fairness = value

        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += int(labels.numel())
        predictions_all.append(predictions.detach().cpu())
        labels_all.append(labels.detach().cpu())
        sensitive_all.append(sensitive_hard.detach().cpu())
        batch_weight = int(labels.numel())
        sums["loss"] += float(loss.item()) * batch_weight
        sums["task"] += float(loss_task.item()) * batch_weight
        sums["sens"] += float(loss_sensitive.item()) * batch_weight
        sums["sens_acc"] += float(sensitive_accuracy.item()) * batch_weight
        sums["cost"] += float(loss_cost.item()) * batch_weight
        sums["fairness"] += float(fairness.detach().item()) * batch_weight
        sums["qset"] += float(loss_qset.item()) * batch_weight
        sums["query_entropy"] += (
            float(torch.stack(entropy_terms).mean().item()) * batch_weight if entropy_terms else 0.0
        )
        sums["set_query_rate"] += (
            float(torch.stack(set_rate_terms).mean().item()) * batch_weight if set_rate_terms else 0.0
        )
        sums["realized_leakage"] += float(total_leakage.mean().item()) * batch_weight
        sums["mean_queries"] += float(query_count.mean().item()) * batch_weight
        n_batches += 1

    predictions = torch.cat(predictions_all)
    labels = torch.cat(labels_all)
    sensitive = torch.cat(sensitive_all)
    result = {
        "acc": correct / max(total, 1),
        "macro_f1": float(macro_f1_score(predictions, labels, classifier.output_dim).item()),
        "eo_violation": float(hard_equalized_odds_violation(
            predictions,
            labels,
            sensitive,
            num_label_classes=classifier.output_dim,
            num_sensitive_classes=objective.num_sensitive_classes,
        ).item()),
        **{
            name: value / max(n_batches if name == "actor_grad" else total, 1)
            for name, value in sums.items()
        },
        "lambda_fairness": float(objective.lambda_fairness),
    }
    return result


def fit_claq_tensor(
    *,
    actor,
    classifier,
    s_head,
    optimizer,
    train_loader,
    validation_loader,
    objective_config: ClaqObjectiveConfig,
    history_config: HistorySamplingConfig,
    device: torch.device,
    num_epochs: int,
    actor_eps_end: float | None = None,
    selection_metric: str = "acc",
    selection_mode: str = "max",
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
) -> tuple[list[dict], dict]:
    """Fit precomputed-response CLAQ and select a checkpoint on validation data."""

    if num_epochs < 1:
        raise ValueError("num_epochs must be at least 1")
    if selection_mode not in {"max", "min"}:
        raise ValueError("selection_mode must be 'max' or 'min'")
    if actor_eps_end is not None and actor_eps_end <= 0:
        raise ValueError("actor_eps_end must be positive")

    history = []
    best_score = float("-inf") if selection_mode == "max" else float("inf")
    best = {"validation_score": best_score}
    eps_start = actor.eps
    for epoch in tqdm(range(1, num_epochs + 1), desc="Tensor CLAQ epochs"):
        if actor_eps_end is not None:
            progress = 0.0 if num_epochs <= 1 else (epoch - 1) / (num_epochs - 1)
            actor.change_eps(float(eps_start + (actor_eps_end - eps_start) * progress))
        train_metrics = run_claq_tensor_epoch(
            loader=train_loader,
            actor=actor,
            classifier=classifier,
            s_head=s_head,
            optimizer=optimizer,
            objective_config=objective_config,
            history_config=history_config,
            device=device,
            train=True,
            max_batches=max_train_batches,
        )
        validation_metrics = run_claq_tensor_epoch(
            loader=validation_loader,
            actor=actor,
            classifier=classifier,
            s_head=s_head,
            optimizer=None,
            objective_config=objective_config,
            history_config=history_config,
            device=device,
            train=False,
            max_batches=max_validation_batches,
        )
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"validation_{k}": v for k, v in validation_metrics.items()}}
        history.append(row)
        if selection_metric not in validation_metrics:
            raise KeyError(
                f"Unknown selection metric '{selection_metric}'. Available metrics: "
                f"{sorted(validation_metrics)}"
            )
        score = float(validation_metrics[selection_metric])
        if not torch.isfinite(torch.tensor(score)):
            raise ValueError(f"Validation selection metric '{selection_metric}' is not finite")
        improved = score >= best["validation_score"] if selection_mode == "max" else score <= best["validation_score"]
        if improved:
            best = {
                "validation_score": score,
                "epoch": epoch,
                "actor_state_dict": _cpu_state_dict(actor),
                "classifier_state_dict": _cpu_state_dict(classifier),
                "s_head_state_dict": _cpu_state_dict(s_head),
                "history_row": row,
                "actor_eps": float(actor.eps),
                "lambda_fairness": float(objective_config.lambda_fairness),
            }
    return history, best


def evaluate_claq_tensor(**kwargs) -> dict:
    """Evaluate a selected tensor-response CLAQ checkpoint."""

    kwargs = dict(kwargs)
    kwargs["train"] = False
    kwargs["optimizer"] = None
    return run_claq_tensor_epoch(**kwargs)
