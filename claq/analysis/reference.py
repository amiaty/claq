"""Build training-reference information scorers and held-out rollout datasets."""

from __future__ import annotations

import torch
from tqdm.auto import tqdm

from claq.constraints import EmpiricalConditionalMIScorer
from claq.core.clip_features import encode_images
from claq.core.runtime import concept_answers_from_image_features, rollout_claq_batch
from claq.sensitive_labels import compute_s_from_concept_targets, compute_s_from_image_features


def _unpack(batch):
    if len(batch) == 2:
        return batch[0], batch[1], None, None
    if len(batch) == 3:
        return batch[0], batch[1], batch[2], None
    if len(batch) == 4:
        return batch[0], batch[1], batch[2], batch[3]
    raise ValueError("Expected a two-, three-, or four-element CLAQ batch")


@torch.no_grad()
def collect_response_reference(
    *,
    loader,
    model_clip,
    dictionary: torch.Tensor,
    answering_model,
    sensitive_indices: torch.Tensor,
    clip_device: torch.device,
    train_device: torch.device,
    threshold_for_binarization: float,
    sensitive_tau: float = 0.7,
    sensitive_topk: int = 3,
    max_batches: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect complete predicted responses, labels, and hard S on a training split."""

    answer_chunks = []
    label_chunks = []
    sensitive_chunks = []
    for batch_index, batch in enumerate(tqdm(loader, desc="Collecting MI reference", leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, labels, concept_targets, direct_sensitive = _unpack(batch)
        image_features = encode_images(model_clip=model_clip, images=images, device=clip_device)
        answers = concept_answers_from_image_features(
            image_features=image_features,
            dictionary=dictionary,
            answering_model=answering_model,
            train_device=train_device,
            threshold=threshold_for_binarization,
        )
        if direct_sensitive is not None:
            sensitive_hard = direct_sensitive.to(torch.long).view(-1)
        elif concept_targets is None:
            _, sensitive_hard = compute_s_from_image_features(
                image_features=image_features,
                logit_scale=model_clip.logit_scale.exp(),
                dictionary=dictionary,
                sens_idx=sensitive_indices,
                tau=sensitive_tau,
                topk=sensitive_topk,
            )
        else:
            _, sensitive_hard = compute_s_from_concept_targets(
                concept_targets=concept_targets.to(train_device),
                sens_idx=sensitive_indices,
            )
        answer_chunks.append(answers.detach().cpu())
        label_chunks.append(labels.detach().cpu().to(torch.long))
        sensitive_chunks.append(sensitive_hard.detach().cpu().to(torch.long))
    if not answer_chunks:
        raise ValueError("The reference loader produced no examples")
    return torch.cat(answer_chunks), torch.cat(label_chunks), torch.cat(sensitive_chunks)


def build_mi_scorer_from_loader(
    *,
    alpha: float = 0.5,
    min_support: int = 32,
    **collect_kwargs,
) -> EmpiricalConditionalMIScorer:
    answers, labels, sensitive = collect_response_reference(**collect_kwargs)
    return EmpiricalConditionalMIScorer(
        answers=answers,
        labels=labels,
        sensitive=sensitive,
        alpha=alpha,
        min_support=min_support,
    )


@torch.no_grad()
def collect_rollout_dataset(
    *,
    loader,
    actor,
    classifier,
    model_clip,
    dictionary: torch.Tensor,
    answering_model,
    sensitive_indices: torch.Tensor,
    clip_device: torch.device,
    train_device: torch.device,
    threshold_for_binarization: float,
    max_steps: int,
    constraint_config=None,
    mi_scorer=None,
    confidence_threshold: float | None = None,
    sensitive_tau: float = 0.7,
    sensitive_topk: int = 3,
    max_batches: int | None = None,
) -> dict[str, torch.Tensor | list[str]]:
    """Collect terminal transcripts and labels for held-out metrics/probes."""

    states = []
    labels_all = []
    sensitive_all = []
    predictions = []
    query_counts = []
    costs = []
    leakages = []
    stop_reasons: list[str] = []
    for batch_index, batch in enumerate(tqdm(loader, desc="Collecting CLAQ rollouts", leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, labels, concept_targets, direct_sensitive = _unpack(batch)
        image_features = encode_images(model_clip=model_clip, images=images, device=clip_device)
        answers = concept_answers_from_image_features(
            image_features=image_features,
            dictionary=dictionary,
            answering_model=answering_model,
            train_device=train_device,
            threshold=threshold_for_binarization,
        )
        if direct_sensitive is not None:
            sensitive_hard = direct_sensitive.to(torch.long).view(-1)
        elif concept_targets is None:
            _, sensitive_hard = compute_s_from_image_features(
                image_features=image_features,
                logit_scale=model_clip.logit_scale.exp(),
                dictionary=dictionary,
                sens_idx=sensitive_indices,
                tau=sensitive_tau,
                topk=sensitive_topk,
            )
        else:
            _, sensitive_hard = compute_s_from_concept_targets(
                concept_targets=concept_targets.to(train_device),
                sens_idx=sensitive_indices,
            )
        result = rollout_claq_batch(
            answers=answers,
            actor=actor,
            classifier=classifier,
            max_steps=max_steps,
            constraint_config=constraint_config,
            mi_scorer=mi_scorer,
            confidence_threshold=confidence_threshold,
        )
        states.append(result["terminal_state"].cpu())
        labels_all.append(labels.to(torch.long).cpu())
        sensitive_all.append(sensitive_hard.to(torch.long).cpu())
        predictions.append(result["predictions"].cpu())
        query_counts.append(result["query_counts"].cpu())
        costs.append(result["cumulative_cost"].cpu())
        leakages.append(result["cumulative_leakage"].cpu())
        stop_reasons.extend(result["stop_reason"])
    return {
        "transcripts": torch.cat(states),
        "labels": torch.cat(labels_all),
        "sensitive": torch.cat(sensitive_all),
        "predictions": torch.cat(predictions),
        "query_counts": torch.cat(query_counts),
        "cumulative_cost": torch.cat(costs),
        "cumulative_leakage": torch.cat(leakages),
        "stop_reason": stop_reasons,
    }
