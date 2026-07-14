"""Compact, versioned checkpoint helpers for CLAQ policy runs."""

from __future__ import annotations

from pathlib import Path

import torch

from claq.models import ConceptNet2, Network

POLICY_CHECKPOINT_VERSION = 1


def load_concept_qa_checkpoint(checkpoint_path: str | Path, device: torch.device) -> ConceptNet2:
    model = ConceptNet2().to(device)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def save_bundle_checkpoint(
    checkpoint_path: str | Path,
    actor: Network | None = None,
    classifier: Network | None = None,
    s_head: Network | None = None,
    optimizer=None,
    metadata: dict | None = None,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {} if metadata is None else dict(metadata)
    payload.setdefault("format_version", POLICY_CHECKPOINT_VERSION)
    payload.setdefault("kind", "claq_policy")
    if actor is not None:
        payload["actor_state_dict"] = actor.state_dict()
        payload.setdefault("max_queries", actor.query_size)
        payload.setdefault("actor_eps", actor.eps)
    if classifier is not None:
        payload["classifier_state_dict"] = classifier.state_dict()
        payload.setdefault("num_classes", classifier.output_dim)
    if s_head is not None:
        payload["s_head_state_dict"] = s_head.state_dict()
        max_queries = payload.get("max_queries")
        if max_queries is not None:
            payload.setdefault(
                "sensitive_conditioning",
                "conditional_y" if s_head.query_size > int(max_queries) else "unconditional",
            )
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, checkpoint_path)


def load_vip_bundle(
    checkpoint_path: str | Path,
    device: torch.device,
    max_queries: int = 128,
    num_classes: int = 10,
    actor_eps: float = 1.0,
):
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    stored_max_queries = int(ckpt.get("max_queries", max_queries))
    stored_num_classes = int(ckpt.get("num_classes", num_classes))
    if stored_max_queries != max_queries or stored_num_classes != num_classes:
        raise ValueError(
            "Checkpoint architecture mismatch: "
            f"stored ({stored_max_queries} queries, {stored_num_classes} classes), "
            f"requested ({max_queries} queries, {num_classes} classes)."
        )
    actor = Network(query_size=max_queries, output_size=max_queries, eps=actor_eps).to(device)
    classifier = Network(query_size=max_queries, output_size=num_classes, eps=None).to(device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    classifier.load_state_dict(ckpt["classifier_state_dict"])
    s_head_state = ckpt.get("s_head_state_dict", ckpt.get("sensitive_head_state_dict"))
    s_head = None
    if s_head_state is not None:
        input_size = int(s_head_state["layer1.weight"].shape[1])
        expected_sizes = {max_queries, max_queries + num_classes}
        if input_size not in expected_sizes:
            raise ValueError(
                f"Sensitive-head input size {input_size} is incompatible with "
                f"{max_queries} queries and {num_classes} classes."
            )
        s_head = Network(query_size=input_size, output_size=1, eps=None).to(device)
        s_head.load_state_dict(s_head_state)
        s_head.eval()
    actor.eval()
    classifier.eval()
    return {
        "ckpt_path": checkpoint_path,
        "meta": ckpt,
        "actor": actor,
        "classifier": classifier,
        "s_head": s_head,
    }


def load_run_bundle(
    checkpoint_path: str | Path,
    device: torch.device,
    max_queries: int = 128,
    num_classes: int = 10,
    actor_eps: float = 1.0,
):
    return load_vip_bundle(
        checkpoint_path=checkpoint_path,
        device=device,
        max_queries=max_queries,
        num_classes=num_classes,
        actor_eps=actor_eps,
    )
