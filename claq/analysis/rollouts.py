"""Tiny-start rollout helpers adapted from the CLAQ analysis notebook work."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def build_random_initial_history(
    answers: torch.Tensor,
    sample_idx: int,
    trial: int,
    min_history: int = 1,
    max_history: int = 2,
    mode: str = "random",
    sensitive_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    num_queries = answers.size(1)
    rng = np.random.default_rng(1000003 + int(sample_idx) + 100000 * int(trial))

    if mode == "random":
        pool = list(range(num_queries))
    elif mode == "non_sensitive":
        if sensitive_indices is None:
            raise ValueError("sensitive_indices are required for non_sensitive mode")
        sensitive_set = set(sensitive_indices.tolist())
        pool = [idx for idx in range(num_queries) if idx not in sensitive_set]
    else:
        raise ValueError(f"Unknown history mode: {mode}")

    if not pool:
        chosen = []
    else:
        capped_max = min(max_history, len(pool))
        capped_min = min(min_history, capped_max)
        history_size = int(rng.integers(low=capped_min, high=capped_max + 1))
        chosen = [] if history_size == 0 else rng.choice(pool, size=history_size, replace=False).tolist()

    mask = torch.zeros_like(answers)
    if chosen:
        mask[0, torch.tensor(chosen, device=answers.device, dtype=torch.long)] = 1.0
    return mask, answers * mask, chosen


def first_divergence_step(seq_a: list[dict], seq_b: list[dict]) -> int | None:
    shared_steps = min(len(seq_a), len(seq_b))
    for idx in range(shared_steps):
        if seq_a[idx]["idx"] != seq_b[idx]["idx"]:
            return idx + 1
    if len(seq_a) != len(seq_b):
        return shared_steps + 1
    return None


def rollout_until_confidence(
    bundle: dict,
    answers_row: torch.Tensor,
    init_mask: torch.Tensor,
    concepts: list[str],
    sensitive_mask: torch.Tensor,
    class_names: list[str],
    threshold: float = 0.95,
    max_steps: int = 20,
    positive_class_idx: int | None = None,
    positive_class_name: str | None = None,
) -> dict:
    mask = init_mask.clone()
    masked_answers = answers_row.unsqueeze(0) * mask
    states = []
    sequence = []
    stop_reason = "max_steps"
    empty_masked_answers = torch.zeros_like(masked_answers)
    empty_logits = bundle["classifier"](empty_masked_answers)
    empty_probs = F.softmax(empty_logits, dim=1)
    empty_conf, empty_pred = empty_probs.max(dim=1)
    empty_pred_idx = int(empty_pred[0].item())
    empty_conf_value = float(empty_conf[0].item())
    empty_positive_prob = None
    if positive_class_idx is not None:
        empty_positive_prob = float(empty_probs[0, positive_class_idx].item())

    for step in range(max_steps + 1):
        logits = bundle["classifier"](masked_answers)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        pred_idx = int(pred[0].item())
        conf_value = float(conf[0].item())
        state = {
            "after_queries": int(step),
            "pred_idx": pred_idx,
            "pred_name": class_names[pred_idx],
            "confidence": conf_value,
        }
        if positive_class_idx is not None:
            state["positive_prob"] = float(probs[0, positive_class_idx].item())
        states.append(state)

        if conf_value >= threshold:
            stop_reason = "confidence"
            break
        if not bool((mask[0] < 0.5).any().item()):
            stop_reason = "all_known"
            break
        if step == max_steps:
            break

        query_distribution = bundle["actor"](masked_answers, mask)[0].clone()
        query_distribution = query_distribution.masked_fill(mask[0] > 0.5, -1e9)
        query_idx = int(query_distribution.argmax().item())
        answer_val = float(answers_row[query_idx].item())
        sequence.append(
            {
                "step": step + 1,
                "idx": query_idx,
                "concept": concepts[query_idx],
                "answer": answer_val,
                "answer_name": "yes" if answer_val > 0 else "no",
                "sensitive": bool(sensitive_mask[query_idx].item() > 0.5),
                "prob": float(query_distribution[query_idx].item()),
            }
        )
        mask[0, query_idx] = 1.0
        masked_answers[0, query_idx] = answers_row[query_idx]

    first_sensitive = next((item["step"] for item in sequence if item["sensitive"]), None)
    result = {
        "states": states,
        "sequence": sequence,
        "queries_asked": int(len(sequence)),
        "sensitive_steps": int(sum(int(item["sensitive"]) for item in sequence)),
        "first_sensitive_step": None if first_sensitive is None else int(first_sensitive),
        "stop_reason": stop_reason,
        "reached_threshold": bool(states[-1]["confidence"] >= threshold),
        "final_pred_idx": int(states[-1]["pred_idx"]),
        "final_pred_name": states[-1]["pred_name"],
        "final_confidence": float(states[-1]["confidence"]),
        "initial_confidence": float(states[0]["confidence"]),
        "initial_pred_name": states[0]["pred_name"],
        "empty_confidence": empty_conf_value,
        "empty_pred_idx": empty_pred_idx,
        "empty_pred_name": class_names[empty_pred_idx],
    }
    if positive_class_idx is not None:
        result["positive_class_idx"] = int(positive_class_idx)
        result["positive_class_name"] = positive_class_name or class_names[positive_class_idx]
        result["empty_positive_prob"] = float(empty_positive_prob)
        result["initial_positive_prob"] = float(states[0]["positive_prob"])
        result["final_positive_prob"] = float(states[-1]["positive_prob"])
    return result


def format_stop_sequence(sequence: list[dict], max_items: int | None = None) -> str:
    if not sequence:
        return "(stops immediately)"
    items = sequence if max_items is None else sequence[:max_items]
    parts = []
    for item in items:
        suffix = " [S]" if item["sensitive"] else ""
        parts.append(f"{item['step']}. {item['concept']}={item['answer_name']}{suffix}")
    if max_items is not None and len(sequence) > max_items:
        parts.append("...")
    return " -> ".join(parts)


def format_confidence_path(states: list[dict], max_items: int | None = None, key: str = "confidence") -> str:
    if not states:
        return "(no confidence states)"
    items = states if max_items is None else states[:max_items]
    parts = [f"{item['after_queries']}:{item[key]:.2f}" for item in items]
    if max_items is not None and len(states) > max_items:
        parts.append("...")
    return " -> ".join(parts)
