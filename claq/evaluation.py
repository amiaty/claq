"""Held-out leakage probes and multi-seed aggregation for CLAQ."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


@dataclass(frozen=True)
class ProbeLeakageResult:
    conditional_entropy_baseline_bits: float
    probe_cross_entropy_bits: float
    probe_leakage_lower_bound_bits: float
    probe_accuracy: float


def empirical_conditional_entropy_bits(sensitive: np.ndarray, labels: np.ndarray) -> float:
    """Plug-in H(S|Y) in bits for discrete arrays."""

    sensitive = np.asarray(sensitive, dtype=int).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if sensitive.shape != labels.shape:
        raise ValueError("sensitive and labels must have the same shape")
    total = len(labels)
    if total == 0:
        return float("nan")
    entropy = 0.0
    for y_value in np.unique(labels):
        idx = labels == y_value
        counts = np.bincount(sensitive[idx])
        probabilities = counts[counts > 0] / counts.sum()
        entropy_y = -(probabilities * np.log2(probabilities)).sum()
        entropy += idx.mean() * entropy_y
    return float(entropy)


def _augment_with_label(transcripts: np.ndarray, labels: np.ndarray, num_label_classes: int) -> np.ndarray:
    one_hot = np.eye(num_label_classes, dtype=np.float64)[np.asarray(labels, dtype=int)]
    return np.concatenate([np.asarray(transcripts, dtype=np.float64), one_hot], axis=1)


def fit_conditional_leakage_probe(
    *,
    train_transcripts: np.ndarray,
    train_labels: np.ndarray,
    train_sensitive: np.ndarray,
    validation_transcripts: np.ndarray,
    validation_labels: np.ndarray,
    validation_sensitive: np.ndarray,
    test_transcripts: np.ndarray,
    test_labels: np.ndarray,
    test_sensitive: np.ndarray,
    c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
    max_iter: int = 2000,
    random_state: int = 0,
) -> ProbeLeakageResult:
    """Fit an independent S|(H,Y) logistic probe and evaluate on test data.

    Hyperparameter C is selected by validation cross-entropy.  The returned
    leakage value is max{0, H_hat(S|Y) - CE_probe}; it is a model-dependent
    lower bound, not an exact mutual-information estimate.
    """

    train_labels = np.asarray(train_labels, dtype=int)
    validation_labels = np.asarray(validation_labels, dtype=int)
    test_labels = np.asarray(test_labels, dtype=int)
    train_sensitive = np.asarray(train_sensitive, dtype=int)
    validation_sensitive = np.asarray(validation_sensitive, dtype=int)
    test_sensitive = np.asarray(test_sensitive, dtype=int)
    train_label_classes = np.unique(train_labels)
    unseen_validation_labels = np.setdiff1d(np.unique(validation_labels), train_label_classes)
    unseen_test_labels = np.setdiff1d(np.unique(test_labels), train_label_classes)
    if unseen_validation_labels.size or unseen_test_labels.size:
        raise ValueError(
            "Every task-label class in validation and test must occur in the probe-training split; "
            f"unseen validation={unseen_validation_labels.tolist()}, unseen test={unseen_test_labels.tolist()}"
        )
    num_label_classes = int(train_labels.max() + 1)
    sensitive_classes = np.unique(train_sensitive)
    unseen_validation = np.setdiff1d(np.unique(validation_sensitive), sensitive_classes)
    unseen_test = np.setdiff1d(np.unique(test_sensitive), sensitive_classes)
    if unseen_validation.size or unseen_test.size:
        raise ValueError(
            "Every sensitive class in validation and test must occur in the probe-training split; "
            f"unseen validation={unseen_validation.tolist()}, unseen test={unseen_test.tolist()}"
        )
    if sensitive_classes.size < 2:
        raise ValueError("The probe-training split must contain at least two sensitive classes")

    x_train = _augment_with_label(train_transcripts, train_labels, num_label_classes)
    x_validation = _augment_with_label(validation_transcripts, validation_labels, num_label_classes)
    x_test = _augment_with_label(test_transcripts, test_labels, num_label_classes)

    best_model = None
    best_validation_ce = float("inf")
    for c_value in c_grid:
        model = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(
                C=float(c_value),
                max_iter=max_iter,
                random_state=random_state,
                class_weight="balanced",
            ),
        )
        model.fit(x_train, train_sensitive)
        validation_probabilities = model.predict_proba(x_validation)
        ce = log_loss(
            validation_sensitive,
            validation_probabilities,
            labels=sensitive_classes,
        ) / np.log(2.0)
        if ce < best_validation_ce:
            best_validation_ce = float(ce)
            best_model = model
    if best_model is None:
        raise RuntimeError("Probe model selection failed")

    test_probabilities = best_model.predict_proba(x_test)
    test_predictions = best_model.predict(x_test)
    test_ce_bits = float(
        log_loss(test_sensitive, test_probabilities, labels=sensitive_classes) / np.log(2.0)
    )
    baseline_entropy = empirical_conditional_entropy_bits(test_sensitive, test_labels)
    return ProbeLeakageResult(
        conditional_entropy_baseline_bits=baseline_entropy,
        probe_cross_entropy_bits=test_ce_bits,
        probe_leakage_lower_bound_bits=max(0.0, baseline_entropy - test_ce_bits),
        probe_accuracy=float(accuracy_score(test_sensitive, test_predictions)),
    )


def aggregate_seed_metrics(rows: list[dict], metric_names: list[str]) -> dict[str, dict[str, float]]:
    """Return mean, sample standard deviation, and count for each metric."""

    result: dict[str, dict[str, float | int]] = {}
    for metric in metric_names:
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        result[metric] = {
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0 if finite.size == 1 else float("nan"),
            "n": int(finite.size),
            "n_total": int(values.size),
        }
    return result
