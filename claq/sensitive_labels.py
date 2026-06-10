"""Sensitive-label helpers adapted from the CLAQ notebook workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from claq.core.clip_features import compute_similarity_scores, encode_images

CIFAR10_SENSITIVE_CONCEPTS = [
    "a bridle",
    "a cab for the driver",
    "a captain",
    "a collar",
    "a copilot",
    "a dashboard",
    "a driver",
    "a flight attendant",
    "a gear shift",
    "a halter",
    "a hitch",
    "a lead rope",
    "a leash",
    "a passenger",
    "a pedal",
    "a pilot",
    "a reins",
    "a rider",
    "a rifle",
    "a saddle",
    "a seatbelt",
    "a steering wheel",
    "a trailer",
]


@dataclass
class SensitiveConceptMatch:
    indices: torch.Tensor
    matched: list[str]
    missing: list[str]


def build_sensitive_index_from_patterns(concepts: list[str], patterns: list[str]) -> torch.Tensor:
    return torch.tensor(
        [idx for idx, concept in enumerate(concepts) if any(pattern in concept.lower() for pattern in patterns)],
        dtype=torch.long,
    )


def match_exact_sensitive_concepts(concepts: list[str], selected_concepts: list[str]) -> SensitiveConceptMatch:
    lookup = {concept.lower(): idx for idx, concept in enumerate(concepts)}
    matched = [concept for concept in selected_concepts if concept.lower() in lookup]
    missing = [concept for concept in selected_concepts if concept.lower() not in lookup]
    indices = torch.tensor([lookup[concept.lower()] for concept in matched], dtype=torch.long)
    return SensitiveConceptMatch(indices=indices, matched=matched, missing=missing)


def build_cifar10_sensitive_match(concepts: list[str]) -> SensitiveConceptMatch:
    return match_exact_sensitive_concepts(concepts, CIFAR10_SENSITIVE_CONCEPTS)


@torch.no_grad()
def compute_s_from_image_features(
    image_features: torch.Tensor,
    logit_scale,
    dictionary: torch.Tensor,
    sens_idx: torch.Tensor,
    tau: float = 0.7,
    topk: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    sims = compute_similarity_scores(
        image_features=image_features,
        dictionary=dictionary,
        logit_scale=logit_scale,
    )
    sens_scores = sims[:, sens_idx.to(sims.device)]
    k = min(topk, sens_scores.size(1))
    s_soft = sens_scores.topk(k=k, dim=1).values.mean(dim=1)
    s_hard = (s_soft >= tau).float()
    return s_soft, s_hard


@torch.no_grad()
def compute_s_from_concept_targets(
    concept_targets: torch.Tensor,
    sens_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sens_idx.numel() == 0:
        zeros = torch.zeros(concept_targets.size(0), dtype=torch.float32, device=concept_targets.device)
        return zeros, zeros
    sens_scores = concept_targets[:, sens_idx.to(concept_targets.device)].float()
    s_soft = sens_scores.mean(dim=1)
    s_hard = (sens_scores.max(dim=1).values > 0.5).float()
    return s_soft, s_hard


@torch.no_grad()
def compute_s_batch(
    images: torch.Tensor,
    model_clip,
    dictionary: torch.Tensor,
    sens_idx: torch.Tensor,
    clip_device: torch.device,
    tau: float = 0.7,
    topk: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_features = encode_images(model_clip=model_clip, images=images, device=clip_device)
    return compute_s_from_image_features(
        image_features=image_features,
        logit_scale=model_clip.logit_scale.exp(),
        dictionary=dictionary,
        sens_idx=sens_idx,
        tau=tau,
        topk=topk,
    )


def build_sensitive_labels_from_concept_targets(
    concept_targets: torch.Tensor,
    sens_idx: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    s_soft, s_hard = compute_s_from_concept_targets(
        concept_targets=concept_targets,
        sens_idx=sens_idx,
    )
    return s_soft.cpu().numpy(), s_hard.cpu().numpy()


def build_sensitive_labels(
    loader,
    model_clip,
    dictionary: torch.Tensor,
    sens_idx: torch.Tensor,
    clip_device: torch.device,
    tau: float = 0.7,
    topk: int = 3,
    desc: str = "Building sensitive labels",
) -> tuple[np.ndarray, np.ndarray]:
    soft_chunks = []
    hard_chunks = []
    for images, _ in tqdm(loader, desc=desc, leave=False):
        s_soft, s_hard = compute_s_batch(
            images=images,
            model_clip=model_clip,
            dictionary=dictionary,
            sens_idx=sens_idx,
            clip_device=clip_device,
            tau=tau,
            topk=topk,
        )
        soft_chunks.append(s_soft.cpu())
        hard_chunks.append(s_hard.cpu())
    return torch.cat(soft_chunks).numpy(), torch.cat(hard_chunks).numpy()


def save_sensitive_labels(output_dir: str | Path, train_soft, train_hard, test_soft, test_hard) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "s_soft_train.npy", train_soft)
    np.save(output_dir / "s_hard_train.npy", train_hard)
    np.save(output_dir / "s_soft_test.npy", test_soft)
    np.save(output_dir / "s_hard_test.npy", test_hard)


def load_sensitive_labels(output_dir: str | Path) -> dict[str, np.ndarray]:
    output_dir = Path(output_dir)
    return {
        "s_soft_train": np.load(output_dir / "s_soft_train.npy"),
        "s_hard_train": np.load(output_dir / "s_hard_train.npy"),
        "s_soft_test": np.load(output_dir / "s_soft_test.npy"),
        "s_hard_test": np.load(output_dir / "s_hard_test.npy"),
    }
