"""Compact disk caches for deterministic Concept-QA answers."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ANSWER_CACHE_VERSION = 1


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_answer_cache(
    path: str | Path,
    *,
    answers: torch.Tensor,
    labels: torch.Tensor,
    sensitive_targets: torch.Tensor,
    metadata: Mapping | None = None,
) -> Path:
    """Store hard -1/+1 answers compactly as signed bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    answers = torch.as_tensor(answers).cpu()
    if answers.ndim != 2:
        raise ValueError("answers must have shape [examples, queries]")
    if not torch.all((answers == -1) | (answers == 1)):
        raise ValueError("answer caches only support hard -1/+1 answers")
    labels = torch.as_tensor(labels).long().cpu()
    sensitive_targets = torch.as_tensor(sensitive_targets).cpu()
    if len(answers) != len(labels) or len(answers) != len(sensitive_targets):
        raise ValueError("answers, labels, and sensitive_targets must have equal length")
    torch.save(
        {
            "format_version": ANSWER_CACHE_VERSION,
            "kind": "concept_qa_answers",
            "metadata": dict(metadata or {}),
            "answers": answers.to(torch.int8),
            "labels": labels,
            "sensitive_targets": sensitive_targets,
        },
        path,
    )
    return path


def load_answer_cache(
    path: str | Path,
    *,
    expected_metadata: Mapping | None = None,
) -> dict:
    """Load a cache and reject stale or incompatible metadata."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != ANSWER_CACHE_VERSION:
        raise ValueError(f"Unsupported answer-cache version in {path}")
    if payload.get("kind") != "concept_qa_answers":
        raise ValueError(f"{path} is not a Concept-QA answer cache")
    metadata = payload.get("metadata", {})
    for key, expected in dict(expected_metadata or {}).items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Stale answer cache {path}: metadata {key!r} is "
                f"{metadata.get(key)!r}, expected {expected!r}"
            )
    return payload


class CachedAnswersDataset(Dataset):
    """Dataset whose mapping batches bypass image and Concept-QA inference."""

    def __init__(self, payload: Mapping):
        self.answers = torch.as_tensor(payload["answers"], dtype=torch.int8)
        self.labels = torch.as_tensor(payload["labels"], dtype=torch.long)
        self.sensitive_targets = torch.as_tensor(payload["sensitive_targets"])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "precomputed_answers": self.answers[index],
            "labels": self.labels[index],
            "sensitive_targets": self.sensitive_targets[index],
        }


def make_cached_answer_loader(
    payload: Mapping,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        CachedAnswersDataset(payload),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
