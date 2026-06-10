"""Checkpoint helpers for the clean CLAQ repo."""

from __future__ import annotations

from pathlib import Path

import torch

from claq.models import ConceptNet2, Network


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
    if actor is not None:
        payload["actor_state_dict"] = actor.state_dict()
    if classifier is not None:
        payload["classifier_state_dict"] = classifier.state_dict()
    if s_head is not None:
        payload["s_head_state_dict"] = s_head.state_dict()
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
    actor = Network(query_size=max_queries, output_size=max_queries, eps=actor_eps).to(device)
    classifier = Network(query_size=max_queries, output_size=num_classes, eps=None).to(device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    classifier.load_state_dict(ckpt["classifier_state_dict"])
    actor.eval()
    classifier.eval()
    return {
        "ckpt_path": checkpoint_path,
        "meta": ckpt,
        "actor": actor,
        "classifier": classifier,
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
