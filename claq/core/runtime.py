"""Shared inference/runtime helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from claq.core.clip_features import build_concept_qa_inputs, encode_images


@torch.no_grad()
def concept_answers_from_image_features(
    image_features: torch.Tensor,
    dictionary: torch.Tensor,
    answering_model,
    train_device: torch.device,
    threshold: float = 0.0,
    qa_chunk: int = 4096,
) -> torch.Tensor:
    inputs = build_concept_qa_inputs(image_features=image_features, dictionary=dictionary)
    inputs = inputs.to(next(answering_model.parameters()).device).float()

    flat_logits = []
    for start in range(0, inputs.size(0), qa_chunk):
        flat_logits.append(answering_model(inputs[start : start + qa_chunk]))

    batch_size = image_features.size(0)
    num_queries = dictionary.size(1)
    logits = torch.cat(flat_logits, dim=0).view(batch_size, num_queries)
    answers = torch.where(logits > threshold, torch.ones_like(logits), -torch.ones_like(logits))
    return answers.to(train_device)


@torch.no_grad()
def concept_answers_batch(
    images: torch.Tensor,
    model_clip,
    dictionary: torch.Tensor,
    answering_model,
    clip_device: torch.device,
    train_device: torch.device,
    threshold: float = 0.0,
    qa_chunk: int = 4096,
) -> torch.Tensor:
    image_features = encode_images(model_clip=model_clip, images=images, device=clip_device)
    return concept_answers_from_image_features(
        image_features=image_features,
        dictionary=dictionary,
        answering_model=answering_model,
        train_device=train_device,
        threshold=threshold,
        qa_chunk=qa_chunk,
    )


def make_sensitive_mask(num_queries: int, sensitive_indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(num_queries, device=device)
    if sensitive_indices.numel() > 0:
        mask[sensitive_indices.to(device)] = 1.0
    return mask


def apply_query_distribution(masked_answers: torch.Tensor, answers: torch.Tensor, query_distribution: torch.Tensor) -> torch.Tensor:
    return masked_answers + (query_distribution * answers)


def one_actor_step(answers: torch.Tensor, actor, mask: torch.Tensor | None = None):
    if mask is None:
        mask = torch.zeros_like(answers)
    masked_answers = answers * mask
    query_distribution = actor(masked_answers, mask)
    updated_answers = apply_query_distribution(masked_answers=masked_answers, answers=answers, query_distribution=query_distribution)
    return updated_answers, query_distribution, mask


def classifier_snapshot(classifier, masked_answers: torch.Tensor) -> tuple[int, float]:
    logits = classifier(masked_answers)
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return int(pred[0].item()), float(conf[0].item())
