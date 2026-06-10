"""Baseline and CLAQ training helpers."""

from __future__ import annotations

import gc
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from claq.core.clip_features import encode_images
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
    s_head = Network(query_size=max_queries, output_size=1, eps=None).to(device)

    if actor_checkpoint is not None:
        actor.load_state_dict(torch.load(actor_checkpoint, map_location="cpu"))
    if classifier_checkpoint is not None:
        classifier.load_state_dict(torch.load(classifier_checkpoint, map_location="cpu"))

    return actor, classifier, s_head


def _unpack_claq_batch(batch):
    if len(batch) == 2:
        images, labels = batch
        return images, labels, None
    if len(batch) == 3:
        images, labels, concept_targets = batch
        return images, labels, concept_targets
    raise ValueError("CLAQ batches must contain (images, labels) or (images, labels, concept_targets)")


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
    lambda_adv: float,
    alpha_sens: float,
    sensitive_tau: float,
    sensitive_topk: int,
    train: bool = True,
    max_batches: int | None = None,
    sensitive_target_mode: str = "soft",
    sensitive_target_indices: torch.Tensor | None = None,
):
    crit_task = nn.CrossEntropyLoss()
    crit_sens = nn.BCEWithLogitsLoss()
    sensitive_mask = make_sensitive_mask(actor.output_dim, sens_idx, train_device)

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
    sum_qpen = 0.0
    sum_sens_q_rate = 0.0
    sum_q_entropy = 0.0
    sum_actor_grad = 0.0
    n_batches = 0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, labels, concept_targets = _unpack_claq_batch(batch)

        with torch.no_grad():
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
            query_distribution = actor(masked_answers, mask)
            updated_answers = apply_query_distribution(
                masked_answers=masked_answers,
                answers=answers,
                query_distribution=query_distribution,
            )

            labels = labels.to(train_device)
            logits_cls = classifier(updated_answers)
            loss_task = crit_task(logits_cls, labels)

            s_logits = s_head(GradientReversal.apply(updated_answers, lambda_adv)).squeeze(1)
            loss_sens = crit_sens(s_logits, s_target.to(train_device).float())
            sens_preds = (torch.sigmoid(s_logits) > 0.5).float()
            sens_acc = (sens_preds == s_target.to(train_device).float()).float().mean()
            loss_qpen = (query_distribution * sensitive_mask).sum(dim=1).mean() * alpha_sens
            loss = loss_task + loss_sens + loss_qpen

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
        sum_qpen += float(loss_qpen.item())
        sum_sens_q_rate += float((query_distribution * sensitive_mask).sum(dim=1).mean().item())
        sum_q_entropy += float(q_entropy.item())
        n_batches += 1

    metrics = {
        "acc": correct / max(total, 1),
        "loss": sum_loss / max(n_batches, 1),
        "task": sum_task / max(n_batches, 1),
        "sens": sum_sens / max(n_batches, 1),
        "sens_acc": sum_sens_acc / max(n_batches, 1),
        "qpen": sum_qpen / max(n_batches, 1),
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
    lambda_adv: float,
    alpha_sens: float,
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
            lambda_adv=lambda_adv,
            alpha_sens=alpha_sens,
            sensitive_tau=sensitive_tau,
            sensitive_topk=sensitive_topk,
            train=True,
            max_batches=max_train_batches,
            sensitive_target_mode=sensitive_target_mode,
            sensitive_target_indices=sensitive_target_indices,
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
            lambda_adv=lambda_adv,
            alpha_sens=alpha_sens,
            sensitive_tau=sensitive_tau,
            sensitive_topk=sensitive_topk,
            train=False,
            max_batches=max_test_batches,
            sensitive_target_mode=sensitive_target_mode,
            sensitive_target_indices=sensitive_target_indices,
        )
        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "lambda_adv": lambda_adv,
            "alpha_sens": alpha_sens,
            "train_acc": train_metrics["acc"],
            "train_loss": train_metrics["loss"],
            "train_task": train_metrics["task"],
            "train_sens": train_metrics["sens"],
            "train_sens_acc": train_metrics["sens_acc"],
            "train_qpen": train_metrics["qpen"],
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
            "test_qpen": test_metrics["qpen"],
            "test_sens_q_rate": test_metrics["sens_q_rate"],
            "test_q_entropy": test_metrics["q_entropy"],
        }
        history.append(row)

        if test_metrics["acc"] >= best["test_acc"]:
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
