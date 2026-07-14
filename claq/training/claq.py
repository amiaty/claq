"""Baseline and CLAQ training helpers."""

from __future__ import annotations

import gc
import random
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from claq.core.clip_features import encode_images
from claq.core.cost import build_uniform_cost, expected_query_cost
from claq.core.runtime import apply_query_distribution, concept_answers_from_image_features, make_sensitive_mask
from claq.models import Network
from claq.sensitive_labels import compute_s_from_concept_targets, compute_s_from_image_features
from claq.training.history_sampling import HistorySamplingConfig, sample_history_mask


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_claq_models(
    max_queries: int,
    num_classes: int,
    device: torch.device,
    actor_eps: float = 1.0,
    actor_checkpoint: str | None = None,
    classifier_checkpoint: str | None = None,
):
    actor = Network(query_size=max_queries, output_size=max_queries, eps=actor_eps).to(device)
    classifier = Network(query_size=max_queries, output_size=num_classes, eps=None).to(device)
    # The sensitive predictor estimates P(S | knowledge state, Y), so its input
    # contains the encoded knowledge state and a one-hot task label.
    s_head = Network(query_size=max_queries + num_classes, output_size=1, eps=None).to(device)

    if actor_checkpoint is not None:
        actor.load_state_dict(torch.load(actor_checkpoint, map_location="cpu"))
    if classifier_checkpoint is not None:
        classifier.load_state_dict(torch.load(classifier_checkpoint, map_location="cpu"))

    return actor, classifier, s_head


def _unpack_claq_batch(batch):
    if isinstance(batch, Mapping):
        required = {"precomputed_answers", "labels", "sensitive_targets"}
        missing = required.difference(batch)
        if missing:
            raise ValueError(f"Cached CLAQ batch is missing keys: {sorted(missing)}")
        return (
            None,
            batch["labels"],
            None,
            batch["precomputed_answers"],
            batch["sensitive_targets"],
        )
    if len(batch) == 2:
        images, labels = batch
        return images, labels, None, None, None
    if len(batch) == 3:
        images, labels, concept_targets = batch
        return images, labels, concept_targets, None, None
    raise ValueError(
        "CLAQ batches must contain images or a cached-answer mapping"
    )


def run_claq_epoch(
    loader,
    actor,
    classifier,
    s_head,
    optimizer,
    model_clip,
    dictionary: torch.Tensor,
    answering_model,
    sens_idx: torch.Tensor,
    history_config: HistorySamplingConfig,
    clip_device: torch.device,
    train_device: torch.device,
    threshold_for_binarization: float,
    lambda_s: float,
    lambda_c: float,
    sensitive_tau: float,
    sensitive_topk: int,
    train: bool = True,
    max_batches: int | None = None,
    sensitive_target_mode: str = "soft",
    sensitive_target_indices: torch.Tensor | None = None,
    cost_vector: torch.Tensor | None = None,
    training_rollout_steps: int = 1,
    lambda_q: float = 0.0,
    designated_query_mask: torch.Tensor | None = None,
):
    if training_rollout_steps < 1:
        raise ValueError("training_rollout_steps must be at least 1")
    crit_task = nn.CrossEntropyLoss()
    crit_sens = nn.BCEWithLogitsLoss()
    sensitive_mask = make_sensitive_mask(actor.output_dim, sens_idx, train_device)
    if cost_vector is None:
        cost_vector = build_uniform_cost(actor.output_dim, train_device)
    else:
        cost_vector = cost_vector.to(train_device)
    if designated_query_mask is None:
        designated_query_mask = torch.zeros(actor.output_dim, device=train_device)
    else:
        designated_query_mask = designated_query_mask.to(train_device).float()

    if train:
        actor.train()
        classifier.train()
        s_head.train()
    else:
        actor.eval()
        classifier.eval()
        s_head.eval()

    total = 0
    correct = 0
    sum_loss = 0.0
    sum_task = 0.0
    sum_sens = 0.0
    sum_sens_acc = 0.0
    sum_cost = 0.0
    sum_exp_cost = 0.0
    sum_set = 0.0
    sum_designated_queries = 0.0
    sum_sens_q_rate = 0.0
    sum_q_entropy = 0.0
    sum_actor_grad = 0.0
    n_batches = 0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, labels, concept_targets, cached_answers, cached_sensitive = _unpack_claq_batch(batch)

        with torch.no_grad():
            if cached_answers is not None:
                answers = cached_answers.to(train_device, dtype=torch.float32)
                s_target = cached_sensitive.to(train_device, dtype=torch.float32).reshape(-1)
            else:
                image_features = encode_images(model_clip=model_clip, images=images, device=clip_device)
                target_sens_idx = sens_idx if sensitive_target_indices is None else sensitive_target_indices
                if concept_targets is None:
                    s_soft, s_hard = compute_s_from_image_features(
                        image_features=image_features,
                        logit_scale=model_clip.logit_scale.exp(),
                        dictionary=dictionary,
                        sens_idx=target_sens_idx,
                        tau=sensitive_tau,
                        topk=sensitive_topk,
                    )
                else:
                    s_soft, s_hard = compute_s_from_concept_targets(
                        concept_targets=concept_targets.to(train_device),
                        sens_idx=target_sens_idx,
                    )
                if sensitive_target_mode == "soft":
                    s_target = s_soft
                elif sensitive_target_mode in {"hard", "max"}:
                    s_target = s_hard
                else:
                    raise ValueError(f"Unknown sensitive_target_mode: {sensitive_target_mode}")
                answers = concept_answers_from_image_features(
                    image_features=image_features,
                    dictionary=dictionary,
                    answering_model=answering_model,
                    train_device=train_device,
                    threshold=threshold_for_binarization,
                )

        with torch.set_grad_enabled(train):
            mask, masked_answers = sample_history_mask(
                answers=answers,
                config=history_config,
                sensitive_indices=sens_idx,
            )
            labels = labels.to(train_device)
            cumulative_expected_cost = torch.zeros((), device=train_device)
            cumulative_designated_queries = torch.zeros((), device=train_device)
            cumulative_sensitive_queries = torch.zeros((), device=train_device)
            query_distribution = None
            updated_answers = masked_answers
            prefix_task_losses = []
            rollout_steps = min(training_rollout_steps, actor.output_dim)
            for _ in range(rollout_steps):
                query_distribution = actor(updated_answers, mask)
                cumulative_expected_cost = cumulative_expected_cost + expected_query_cost(
                    query_distribution, cost_vector
                )
                cumulative_designated_queries = cumulative_designated_queries + (
                    query_distribution * designated_query_mask
                ).sum(dim=1).mean()
                cumulative_sensitive_queries = cumulative_sensitive_queries + (
                    query_distribution * sensitive_mask
                ).sum(dim=1).mean()
                updated_answers = apply_query_distribution(
                    masked_answers=updated_answers,
                    answers=answers,
                    query_distribution=query_distribution,
                )
                logits_cls = classifier(updated_answers)
                prefix_task_losses.append(crit_task(logits_cls, labels))
                # The forward value is one-hot. Detaching the history mask keeps
                # the discrete knowledge state while gradients flow through answers.
                mask = torch.clamp(mask + query_distribution.detach(), 0.0, 1.0)

            loss_task = torch.stack(prefix_task_losses).mean()

            reversed_knowledge_state = GradientReversal.apply(updated_answers, lambda_s)
            task_labels = F.one_hot(labels, num_classes=classifier.output_dim).to(
                device=train_device,
                dtype=updated_answers.dtype,
            )
            sensitive_input = torch.cat((reversed_knowledge_state, task_labels), dim=1)
            s_logits = s_head(sensitive_input).squeeze(1)
            loss_sens = crit_sens(s_logits, s_target.to(train_device).float())
            sens_preds = (torch.sigmoid(s_logits) > 0.5).float()
            sens_acc = (sens_preds == s_target.to(train_device).float()).float().mean()
            loss_cost = cumulative_expected_cost * lambda_c
            loss_set = cumulative_designated_queries * lambda_q
            loss = loss_task + loss_sens + loss_cost + loss_set

            if train:
                optimizer.zero_grad()
                loss.backward()
                actor_grad = 0.0
                for param in actor.parameters():
                    if param.grad is not None:
                        actor_grad += param.grad.detach().norm().item()
                sum_actor_grad += actor_grad
                optimizer.step()

        pred = logits_cls.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.size(0))

        q_safe = query_distribution.clamp_min(1e-8)
        q_entropy = -(q_safe * torch.log(q_safe)).sum(dim=1).mean()

        sum_loss += float(loss.item())
        sum_task += float(loss_task.item())
        sum_sens += float(loss_sens.item())
        sum_sens_acc += float(sens_acc.item())
        sum_cost += float(loss_cost.item())
        sum_exp_cost += float(cumulative_expected_cost.item())
        sum_set += float(loss_set.item())
        sum_designated_queries += float(cumulative_designated_queries.item())
        sum_sens_q_rate += float((cumulative_sensitive_queries / rollout_steps).item())
        sum_q_entropy += float(q_entropy.item())
        n_batches += 1

    metrics = {
        "acc": correct / max(total, 1),
        "loss": sum_loss / max(n_batches, 1),
        "task": sum_task / max(n_batches, 1),
        "sens": sum_sens / max(n_batches, 1),
        "sens_acc": sum_sens_acc / max(n_batches, 1),
        "cost": sum_cost / max(n_batches, 1),
        "exp_cost": sum_exp_cost / max(n_batches, 1),
        "set": sum_set / max(n_batches, 1),
        "designated_queries": sum_designated_queries / max(n_batches, 1),
        "sens_q_rate": sum_sens_q_rate / max(n_batches, 1),
        "q_entropy": sum_q_entropy / max(n_batches, 1),
    }
    if train:
        metrics["actor_grad_norm"] = sum_actor_grad / max(n_batches, 1)
    return metrics


def fit_claq(
    actor,
    classifier,
    s_head,
    optimizer,
    train_loader,
    test_loader,
    model_clip,
    dictionary: torch.Tensor,
    answering_model,
    sens_idx: torch.Tensor,
    history_config: HistorySamplingConfig,
    clip_device: torch.device,
    train_device: torch.device,
    threshold_for_binarization: float,
    lambda_s: float,
    lambda_c: float,
    sensitive_tau: float,
    sensitive_topk: int,
    num_epochs: int,
    scheduler=None,
    max_train_batches: int | None = None,
    max_test_batches: int | None = None,
    actor_eps_end: float | None = None,
    actor_eps_anneal_epochs: int | None = None,
    sensitive_target_mode: str = "soft",
    sensitive_target_indices: torch.Tensor | None = None,
    cost_vector: torch.Tensor | None = None,
    training_rollout_steps: int = 1,
    lambda_q: float = 0.0,
    designated_query_mask: torch.Tensor | None = None,
):
    history = []
    best = {"test_acc": -1.0}
    epoch_bar = tqdm(range(1, num_epochs + 1), desc="CLAQ epochs")
    actor_eps_start = getattr(actor, "eps", None)
    anneal_epochs = actor_eps_anneal_epochs or num_epochs

    for epoch in epoch_bar:
        if actor_eps_start is not None and actor_eps_end is not None:
            progress = min(max(epoch - 1, 0), max(anneal_epochs - 1, 1)) / max(anneal_epochs - 1, 1)
            current_actor_eps = float(actor_eps_start + (actor_eps_end - actor_eps_start) * progress)
            actor.change_eps(current_actor_eps)
        else:
            current_actor_eps = actor_eps_start
        train_metrics = run_claq_epoch(
            loader=train_loader,
            actor=actor,
            classifier=classifier,
            s_head=s_head,
            optimizer=optimizer,
            model_clip=model_clip,
            dictionary=dictionary,
            answering_model=answering_model,
            sens_idx=sens_idx,
            history_config=history_config,
            clip_device=clip_device,
            train_device=train_device,
            threshold_for_binarization=threshold_for_binarization,
            lambda_s=lambda_s,
            lambda_c=lambda_c,
            sensitive_tau=sensitive_tau,
            sensitive_topk=sensitive_topk,
            train=True,
            max_batches=max_train_batches,
            sensitive_target_mode=sensitive_target_mode,
            sensitive_target_indices=sensitive_target_indices,
            cost_vector=cost_vector,
            training_rollout_steps=training_rollout_steps,
            lambda_q=lambda_q,
            designated_query_mask=designated_query_mask,
        )
        test_metrics = run_claq_epoch(
            loader=test_loader,
            actor=actor,
            classifier=classifier,
            s_head=s_head,
            optimizer=optimizer,
            model_clip=model_clip,
            dictionary=dictionary,
            answering_model=answering_model,
            sens_idx=sens_idx,
            history_config=history_config,
            clip_device=clip_device,
            train_device=train_device,
            threshold_for_binarization=threshold_for_binarization,
            lambda_s=lambda_s,
            lambda_c=lambda_c,
            sensitive_tau=sensitive_tau,
            sensitive_topk=sensitive_topk,
            train=False,
            max_batches=max_test_batches,
            sensitive_target_mode=sensitive_target_mode,
            sensitive_target_indices=sensitive_target_indices,
            cost_vector=cost_vector,
            training_rollout_steps=training_rollout_steps,
            lambda_q=lambda_q,
            designated_query_mask=designated_query_mask,
        )
        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "lambda_s": lambda_s,
            "lambda_c": lambda_c,
            "lambda_q": lambda_q,
            "train_acc": train_metrics["acc"],
            "train_loss": train_metrics["loss"],
            "train_task": train_metrics["task"],
            "train_sens": train_metrics["sens"],
            "train_sens_acc": train_metrics["sens_acc"],
            "train_cost": train_metrics["cost"],
            "train_exp_cost": train_metrics["exp_cost"],
            "train_set": train_metrics["set"],
            "train_designated_queries": train_metrics["designated_queries"],
            "train_sens_q_rate": train_metrics["sens_q_rate"],
            "train_q_entropy": train_metrics["q_entropy"],
            "train_actor_grad_norm": train_metrics.get("actor_grad_norm"),
            "actor_eps": None if current_actor_eps is None else float(current_actor_eps),
            "sensitive_target_mode": sensitive_target_mode,
            "test_acc": test_metrics["acc"],
            "test_loss": test_metrics["loss"],
            "test_task": test_metrics["task"],
            "test_sens": test_metrics["sens"],
            "test_sens_acc": test_metrics["sens_acc"],
            "test_cost": test_metrics["cost"],
            "test_exp_cost": test_metrics["exp_cost"],
            "test_set": test_metrics["set"],
            "test_designated_queries": test_metrics["designated_queries"],
            "test_sens_q_rate": test_metrics["sens_q_rate"],
            "test_q_entropy": test_metrics["q_entropy"],
        }
        history.append(row)

        if test_metrics["acc"] > best["test_acc"]:
            best = {
                "test_acc": test_metrics["acc"],
                "epoch": epoch,
                "actor_state_dict": {k: v.detach().cpu() for k, v in actor.state_dict().items()},
                "classifier_state_dict": {k: v.detach().cpu() for k, v in classifier.state_dict().items()},
                "s_head_state_dict": {k: v.detach().cpu() for k, v in s_head.state_dict().items()},
                "history_row": row,
            }

        epoch_bar.set_postfix(
            train_acc=f"{train_metrics['acc']:.3f}",
            test_acc=f"{test_metrics['acc']:.3f}",
            test_sens=f"{test_metrics['sens_q_rate']:.3f}",
            eps=None if current_actor_eps is None else f"{current_actor_eps:.2f}",
        )

        gc.collect()
        if train_device.type == "cuda":
            torch.cuda.empty_cache()

    return history, best
