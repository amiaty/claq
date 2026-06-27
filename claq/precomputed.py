"""Utilities for reproducible CLAQ experiments with precomputed responses.

The precomputed format is the preferred paper-facing interface: each split
contains a complete finite-alphabet response vector, a task label, and a
sensitive label.  Policy training then uses exactly the operational hard
responses that are evaluated at test time.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from claq.constraints import EmpiricalConditionalMIScorer
from claq.core.runtime import rollout_claq_batch
from claq.objectives import hard_equalized_odds_details, macro_f1_score


REQUIRED_SPLITS = ("train", "validation", "test")
PRECOMPUTED_FORMAT_VERSION = 1


def normalize_binary_responses(values: torch.Tensor) -> torch.Tensor:
    """Return responses in ``{-1,+1}``, reserving zero for an unqueried entry."""

    values = values.to(torch.float32)
    if values.ndim != 2:
        raise ValueError("responses must have shape [num_examples, num_queries]")
    is_zero_one = bool(((values == 0) | (values == 1)).all().item())
    if is_zero_one:
        values = values * 2.0 - 1.0
    if not bool(((values == -1) | (values == 1)).all().item()):
        raise ValueError("responses must be binary and encoded as {0,1} or {-1,+1}")
    return values


@dataclass(frozen=True)
class PrecomputedSplit:
    responses: torch.Tensor
    labels: torch.Tensor
    sensitive: torch.Tensor

    def validate(self, *, num_queries: int | None = None) -> None:
        if self.responses.ndim != 2:
            raise ValueError("responses must be rank 2")
        if self.responses.size(0) == 0:
            raise ValueError("each split must contain at least one example")
        if self.labels.ndim != 1 or self.sensitive.ndim != 1:
            raise ValueError("labels and sensitive must be rank 1")
        n = self.responses.size(0)
        if self.labels.numel() != n or self.sensitive.numel() != n:
            raise ValueError("responses, labels, and sensitive must have equal length")
        if num_queries is not None and self.responses.size(1) != num_queries:
            raise ValueError("all splits must have the same number of queries")
        if not torch.isfinite(self.responses).all():
            raise ValueError("responses must be finite")
        is_zero_one = bool(((self.responses == 0) | (self.responses == 1)).all().item())
        is_minus_plus = bool(((self.responses == -1) | (self.responses == 1)).all().item())
        if not (is_zero_one or is_minus_plus):
            raise ValueError("responses must be binary and encoded as {0,1} or {-1,+1}")
        for name, values in (("labels", self.labels), ("sensitive", self.sensitive)):
            if not torch.isfinite(values).all():
                raise ValueError(f"{name} must be finite")
            if values.is_floating_point() and not torch.equal(values, values.round()):
                raise ValueError(f"{name} must be integer-valued class indices")
            if (values < 0).any():
                raise ValueError(f"{name} must be nonnegative class indices")

    def dataset(self) -> TensorDataset:
        return TensorDataset(self.responses, self.labels, self.sensitive)


@dataclass(frozen=True)
class PrecomputedDataset:
    train: PrecomputedSplit
    validation: PrecomputedSplit
    test: PrecomputedSplit
    query_names: tuple[str, ...]
    query_set_mask: torch.Tensor
    query_costs: torch.Tensor
    admissible_mask: torch.Tensor
    justification_mask: torch.Tensor
    metadata: dict[str, Any]

    @property
    def num_queries(self) -> int:
        return int(self.train.responses.size(1))

    @property
    def num_label_classes(self) -> int:
        """Number of task classes inferred from the training split only."""
        return int(self.train.labels.max().item() + 1)

    @property
    def num_sensitive_classes(self) -> int:
        """Number of sensitive classes inferred from the training split only."""
        return int(self.train.sensitive.max().item() + 1)

    def validate(self) -> None:
        self.train.validate()
        self.validation.validate(num_queries=self.num_queries)
        self.test.validate(num_queries=self.num_queries)
        if self.num_queries < 1:
            raise ValueError("the query vocabulary must not be empty")
        for name, tensor in (
            ("query_set_mask", self.query_set_mask),
            ("query_costs", self.query_costs),
            ("admissible_mask", self.admissible_mask),
            ("justification_mask", self.justification_mask),
        ):
            if tensor.ndim != 1 or tensor.numel() != self.num_queries:
                raise ValueError(f"{name} must be a length-{self.num_queries} vector")
        for name, tensor in (
            ("query_set_mask", self.query_set_mask),
            ("admissible_mask", self.admissible_mask),
            ("justification_mask", self.justification_mask),
        ):
            if tensor.dtype != torch.bool:
                raise ValueError(f"{name} must have boolean dtype")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")
        if len(self.query_names) != self.num_queries:
            raise ValueError("query_names has the wrong length")
        if any(not name.strip() for name in self.query_names):
            raise ValueError("query_names must be nonempty strings")
        if len(set(self.query_names)) != len(self.query_names):
            raise ValueError("query_names must be unique")
        if not torch.isfinite(self.query_costs).all():
            raise ValueError("query costs must be finite")
        if (self.query_costs < 0).any():
            raise ValueError("query costs must be nonnegative")

        def validate_training_classes(values: torch.Tensor, name: str, *, minimum_classes: int) -> int:
            observed = torch.unique(values.to(torch.long), sorted=True)
            expected = torch.arange(int(observed[-1].item()) + 1, dtype=torch.long)
            if not torch.equal(observed.cpu(), expected):
                raise ValueError(f"training {name} indices must be contiguous and start at zero")
            if observed.numel() < minimum_classes:
                raise ValueError(f"training split must contain at least {minimum_classes} {name} classes")
            return int(observed.numel())

        num_labels = validate_training_classes(self.train.labels, "label", minimum_classes=2)
        num_sensitive = validate_training_classes(self.train.sensitive, "sensitive", minimum_classes=2)
        for split_name, split in (("validation", self.validation), ("test", self.test)):
            if int(split.labels.max().item()) >= num_labels:
                raise ValueError(f"{split_name} contains a task-label class absent from training")
            if int(split.sensitive.max().item()) >= num_sensitive:
                raise ValueError(f"{split_name} contains a sensitive class absent from training")

    def loaders(self, *, batch_size: int, num_workers: int = 0, seed: int = 0) -> tuple[DataLoader, DataLoader, DataLoader]:
        generator = torch.Generator().manual_seed(int(seed))
        train_loader = DataLoader(
            self.train.dataset(), batch_size=batch_size, shuffle=True,
            num_workers=num_workers, generator=generator,
        )
        validation_loader = DataLoader(
            self.validation.dataset(), batch_size=batch_size, shuffle=False,
            num_workers=num_workers,
        )
        test_loader = DataLoader(
            self.test.dataset(), batch_size=batch_size, shuffle=False,
            num_workers=num_workers,
        )
        return train_loader, validation_loader, test_loader


def _array(data: np.lib.npyio.NpzFile, key: str, *, required: bool = True):
    if key in data:
        return data[key]
    if required:
        raise KeyError(f"Missing required array '{key}'")
    return None


def _integer_index_vector(values: np.ndarray, *, name: str) -> torch.Tensor:
    array = np.asarray(values)
    if array.ndim != 1:
        array = array.reshape(-1)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric class indices")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    rounded = np.rint(array)
    if not np.array_equal(array, rounded):
        raise ValueError(f"{name} must contain integer-valued class indices")
    return torch.from_numpy(rounded.astype(np.int64, copy=False)).view(-1)


def _parse_precomputed_archive(data: np.lib.npyio.NpzFile) -> PrecomputedDataset:
    version_array = _array(data, "claq_format_version", required=False)
    if version_array is not None:
        version = int(np.asarray(version_array).item())
        if version != PRECOMPUTED_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported CLAQ precomputed format version {version}; "
                f"expected {PRECOMPUTED_FORMAT_VERSION}"
            )
    splits: dict[str, PrecomputedSplit] = {}
    for split in REQUIRED_SPLITS:
        responses = normalize_binary_responses(torch.from_numpy(_array(data, f"responses_{split}")))
        labels = _integer_index_vector(_array(data, f"y_{split}"), name=f"y_{split}")
        sensitive = _integer_index_vector(_array(data, f"s_{split}"), name=f"s_{split}")
        splits[split] = PrecomputedSplit(responses, labels, sensitive)

    num_queries = int(splits["train"].responses.size(1))
    query_names_array = _array(data, "query_names", required=False)
    query_names = (
        tuple(str(x) for x in query_names_array.tolist())
        if query_names_array is not None
        else tuple(f"q_{i}" for i in range(num_queries))
    )

    def vector(name: str, default: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        value = _array(data, name, required=False)
        return torch.as_tensor(default if value is None else value, dtype=dtype).view(-1)

    metadata: dict[str, Any] = {}
    metadata_json = _array(data, "metadata_json", required=False)
    if metadata_json is not None:
        raw_json = np.asarray(metadata_json).item()
        parsed = json.loads(str(raw_json))
        if not isinstance(parsed, dict):
            raise ValueError("metadata_json must encode a JSON object")
        metadata = parsed
    else:
        # Legacy archives stored a pickled Python dictionary in ``metadata``.
        metadata_obj = _array(data, "metadata", required=False)
        if metadata_obj is not None:
            raw = metadata_obj.item() if np.asarray(metadata_obj).shape == () else metadata_obj.tolist()
            if isinstance(raw, dict):
                metadata = raw

    return PrecomputedDataset(
        train=splits["train"],
        validation=splits["validation"],
        test=splits["test"],
        query_names=query_names,
        query_set_mask=vector("query_set_mask", np.zeros(num_queries), torch.bool),
        query_costs=vector("query_costs", np.ones(num_queries), torch.float32),
        admissible_mask=vector("admissible_mask", np.ones(num_queries), torch.bool),
        justification_mask=vector("justification_mask", np.zeros(num_queries), torch.bool),
        metadata=metadata,
    )


def load_precomputed_dataset(
    path: str | Path,
    *,
    allow_legacy_pickle: bool = False,
) -> PrecomputedDataset:
    """Load the canonical ``.npz`` response format.

    New archives contain only numeric/string arrays and JSON metadata and are
    loaded with NumPy pickling disabled. Historical archives containing object
    arrays can be opened only through the explicit ``allow_legacy_pickle``
    opt-in and must be treated as trusted input.
    """

    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as data:
            result = _parse_precomputed_archive(data)
    except ValueError as exc:
        object_array_error = "Object arrays cannot be loaded when allow_pickle=False" in str(exc)
        if not object_array_error or not allow_legacy_pickle:
            if object_array_error:
                raise ValueError(
                    "The archive contains legacy pickled object arrays. Re-export it with the current "
                    "package or pass allow_legacy_pickle=True only for a trusted archive."
                ) from exc
            raise
        with np.load(path, allow_pickle=True) as data:
            result = _parse_precomputed_archive(data)
    result.validate()
    return result


def save_precomputed_dataset(path: str | Path, dataset: PrecomputedDataset) -> Path:
    dataset.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        metadata_json = json.dumps(dataset.metadata, sort_keys=True)
    except TypeError as exc:
        raise ValueError("metadata must be JSON serializable") from exc
    np.savez_compressed(
        path,
        claq_format_version=np.asarray(PRECOMPUTED_FORMAT_VERSION, dtype=np.int64),
        responses_train=dataset.train.responses.cpu().numpy(),
        y_train=dataset.train.labels.cpu().numpy(),
        s_train=dataset.train.sensitive.cpu().numpy(),
        responses_validation=dataset.validation.responses.cpu().numpy(),
        y_validation=dataset.validation.labels.cpu().numpy(),
        s_validation=dataset.validation.sensitive.cpu().numpy(),
        responses_test=dataset.test.responses.cpu().numpy(),
        y_test=dataset.test.labels.cpu().numpy(),
        s_test=dataset.test.sensitive.cpu().numpy(),
        query_names=np.asarray(dataset.query_names, dtype=np.str_),
        query_set_mask=dataset.query_set_mask.cpu().numpy(),
        query_costs=dataset.query_costs.cpu().numpy(),
        admissible_mask=dataset.admissible_mask.cpu().numpy(),
        justification_mask=dataset.justification_mask.cpu().numpy(),
        metadata_json=np.asarray(metadata_json),
    )
    return path


def build_mi_scorer_from_split(
    split: PrecomputedSplit,
    *,
    num_label_classes: int,
    num_sensitive_classes: int,
    alpha: float = 0.5,
    min_support: int = 32,
) -> EmpiricalConditionalMIScorer:
    return EmpiricalConditionalMIScorer(
        answers=split.responses,
        labels=split.labels,
        sensitive=split.sensitive,
        num_label_classes=num_label_classes,
        num_sensitive_classes=num_sensitive_classes,
        alpha=alpha,
        min_support=min_support,
    )


@torch.no_grad()
def collect_tensor_rollouts(
    *,
    split: PrecomputedSplit,
    actor,
    classifier,
    device: torch.device,
    max_steps: int,
    batch_size: int,
    constraint_config=None,
    mi_scorer=None,
    confidence_threshold: float | None = None,
    minimum_steps: int = 0,
    query_set_mask: torch.Tensor | None = None,
    report_equalized_odds: bool = True,
    equalized_odds_min_group_count: int = 1,
    num_sensitive_classes: int | None = None,
) -> dict[str, Any]:
    """Roll out a selected policy and return exact held-out trajectory metrics."""

    loader = DataLoader(split.dataset(), batch_size=batch_size, shuffle=False)
    chunks: dict[str, list[torch.Tensor]] = {
        "transcripts": [], "labels": [], "sensitive": [], "predictions": [],
        "query_counts": [], "cumulative_cost": [], "cumulative_leakage": [], "actions": [],
    }
    stop_reasons: list[str] = []
    for responses, labels, sensitive in loader:
        responses = normalize_binary_responses(responses).to(device)
        result = rollout_claq_batch(
            answers=responses,
            actor=actor,
            classifier=classifier,
            max_steps=max_steps,
            constraint_config=constraint_config,
            mi_scorer=mi_scorer,
            confidence_threshold=confidence_threshold,
            minimum_steps=minimum_steps,
        )
        chunks["transcripts"].append(result["terminal_state"].cpu())
        chunks["labels"].append(labels.to(torch.long).cpu())
        chunks["sensitive"].append(sensitive.to(torch.long).cpu())
        chunks["predictions"].append(result["predictions"].cpu())
        chunks["query_counts"].append(result["query_counts"].cpu())
        chunks["cumulative_cost"].append(result["cumulative_cost"].cpu())
        chunks["cumulative_leakage"].append(result["cumulative_leakage"].cpu())
        chunks["actions"].append(result["actions"].cpu())
        stop_reasons.extend(result["stop_reason"])

    merged = {name: torch.cat(values, dim=0) for name, values in chunks.items()}
    labels = merged["labels"]
    predictions = merged["predictions"]
    sensitive = merged["sensitive"]
    metrics = {
        "accuracy": float((predictions == labels).to(torch.float32).mean().item()),
        "macro_f1": float(macro_f1_score(predictions, labels, classifier.output_dim).item()),
        "mean_queries": float(merged["query_counts"].to(torch.float32).mean().item()),
        "mean_cost": float(merged["cumulative_cost"].to(torch.float32).mean().item()),
        "mean_empirical_leakage_account": float(merged["cumulative_leakage"].to(torch.float32).mean().item()),
        "num_examples": int(labels.numel()),
    }
    if report_equalized_odds:
        eo = hard_equalized_odds_details(
            predictions,
            labels,
            sensitive,
            num_label_classes=classifier.output_dim,
            num_sensitive_classes=(
                int(num_sensitive_classes)
                if num_sensitive_classes is not None
                else int(sensitive.max().item()) + 1
            ),
            min_group_count=equalized_odds_min_group_count,
        )
        metrics.update({
            "equalized_odds_violation": float(eo["violation"].item()),
            "equalized_odds_minimum_included_cell_count": int(eo["minimum_included_cell_count"]),
            "equalized_odds_valid_group_cells": int(eo["valid_group_cells"]),
            "equalized_odds_valid_group_pairs": int(eo["valid_group_pairs"]),
        })
    if query_set_mask is not None:
        mask = query_set_mask.to(torch.bool).cpu()
        actions = merged["actions"]
        valid = actions >= 0
        selected = torch.zeros_like(valid)
        selected[valid] = mask[actions[valid]]
        denominator = int(valid.sum().item())
        metrics["designated_query_rate"] = float(selected.sum().item() / denominator) if denominator else float("nan")
        metrics["designated_query_count"] = int(selected.sum().item())
        metrics["executed_query_count"] = denominator
    for reason in ("no_feasible_query", "confidence", "max_steps", "all_queries_exhausted"):
        metrics[f"{reason}_rate"] = (
            sum(item == reason for item in stop_reasons) / len(stop_reasons)
            if stop_reasons else float("nan")
        )
    merged["stop_reason"] = stop_reasons
    merged["metrics"] = metrics
    return merged
