"""Independent diagnostic for conditional protected-attribute recoverability."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_C_VALUES = (0.01, 0.1, 1.0, 10.0)


def _as_1d_int(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(array).all() or not np.equal(array, np.round(array)).all():
        raise ValueError(f"{name} must contain finite integer values")
    return array.astype(np.int64, copy=False)


def _probe_features(knowledge_states, labels, num_classes: int) -> np.ndarray:
    states = np.asarray(knowledge_states, dtype=np.float64)
    labels = _as_1d_int(labels, "labels")
    if states.ndim != 2:
        raise ValueError("knowledge_states must have shape [samples, queries]")
    if len(states) != len(labels):
        raise ValueError("knowledge_states and labels must have the same length")
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    if len(labels) and (labels.min() < 0 or labels.max() >= num_classes):
        raise ValueError("labels contain a value outside [0, num_classes)")

    one_hot = np.eye(num_classes, dtype=np.float64)[labels]
    return np.concatenate((states, one_hot), axis=1)


def conditional_entropy_bits(sensitive_targets, labels) -> float:
    """Return the plug-in estimate of H(S | Y) for a binary attribute."""
    sensitive = _as_1d_int(sensitive_targets, "sensitive_targets")
    labels = _as_1d_int(labels, "labels")
    if len(sensitive) != len(labels):
        raise ValueError("sensitive_targets and labels must have the same length")
    if len(sensitive) == 0:
        raise ValueError("at least one sample is required")
    if not np.isin(sensitive, (0, 1)).all():
        raise ValueError("sensitive_targets must be binary")

    entropy = 0.0
    for label in np.unique(labels):
        group = sensitive[labels == label]
        probability = float(group.mean())
        if probability in (0.0, 1.0):
            group_entropy = 0.0
        else:
            group_entropy = -(
                probability * np.log2(probability)
                + (1.0 - probability) * np.log2(1.0 - probability)
            )
        entropy += (len(group) / len(sensitive)) * group_entropy
    return float(entropy)


def fit_conditional_probe(
    *,
    train_states,
    train_labels,
    train_sensitive,
    validation_states,
    validation_labels,
    validation_sensitive,
    test_states,
    test_labels,
    test_sensitive,
    num_classes: int,
    c_values: Iterable[float] = DEFAULT_C_VALUES,
    random_state: int = 0,
) -> dict[str, object]:
    """Fit an L2-logistic probe on (knowledge state, Y) and evaluate on test.

    The inverse-regularization strength is selected only by validation
    cross-entropy. All cross-entropies and the resulting diagnostic are in bits.
    """
    train_sensitive = _as_1d_int(train_sensitive, "train_sensitive")
    validation_sensitive = _as_1d_int(
        validation_sensitive, "validation_sensitive"
    )
    test_sensitive = _as_1d_int(test_sensitive, "test_sensitive")
    if not np.isin(train_sensitive, (0, 1)).all():
        raise ValueError("train_sensitive must be binary")
    if not np.isin(validation_sensitive, (0, 1)).all():
        raise ValueError("validation_sensitive must be binary")
    if not np.isin(test_sensitive, (0, 1)).all():
        raise ValueError("test_sensitive must be binary")
    if len(np.unique(train_sensitive)) != 2:
        raise ValueError("train_sensitive must contain both classes")

    train_x = _probe_features(train_states, train_labels, num_classes)
    validation_x = _probe_features(
        validation_states, validation_labels, num_classes
    )
    test_x = _probe_features(test_states, test_labels, num_classes)
    if len(train_x) != len(train_sensitive):
        raise ValueError("train states and sensitive targets have different lengths")
    if len(validation_x) != len(validation_sensitive):
        raise ValueError(
            "validation states and sensitive targets have different lengths"
        )
    if len(test_x) != len(test_sensitive):
        raise ValueError("test states and sensitive targets have different lengths")

    candidates = tuple(float(value) for value in c_values)
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("c_values must contain positive values")

    best_model = None
    best_c = None
    best_validation_ce = float("inf")
    validation_ce_by_c = {}
    for c_value in candidates:
        model = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(
                C=c_value,
                penalty="l2",
                solver="liblinear",
                max_iter=2_000,
                random_state=random_state,
            ),
        )
        model.fit(train_x, train_sensitive)
        validation_prob = model.predict_proba(validation_x)[:, 1]
        validation_ce = log_loss(
            validation_sensitive,
            validation_prob,
            labels=[0, 1],
        ) / np.log(2.0)
        validation_ce_by_c[c_value] = float(validation_ce)
        if validation_ce < best_validation_ce:
            best_model = model
            best_c = c_value
            best_validation_ce = float(validation_ce)

    test_prob = best_model.predict_proba(test_x)[:, 1]
    test_ce = float(
        log_loss(test_sensitive, test_prob, labels=[0, 1]) / np.log(2.0)
    )
    entropy = conditional_entropy_bits(test_sensitive, test_labels)
    return {
        "selected_c": float(best_c),
        "validation_cross_entropy_bits": best_validation_ce,
        "test_cross_entropy_bits": test_ce,
        "conditional_entropy_bits": entropy,
        "probe_leakage_bits": max(0.0, entropy - test_ce),
        "validation_cross_entropy_bits_by_c": validation_ce_by_c,
        "test_probe_accuracy": float(
            np.mean((test_prob >= 0.5).astype(np.int64) == test_sensitive)
        ),
    }
