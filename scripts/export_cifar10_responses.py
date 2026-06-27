#!/usr/bin/env python3
"""Export CIFAR-10 learned concept responses to the canonical CLAQ .npz format.

The sensitive variable is explicitly constructed as a binary CLIP-derived
indicator from the designated concept set.  This is a controlled concept study,
not a demographic fairness benchmark.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm

from claq.core import (
    build_concept_dictionary,
    concept_answers_from_image_features,
    encode_images,
    load_clip_model,
    load_concept_qa_checkpoint,
    load_concepts,
)
from claq.data import get_cifar10_train_valid_test_loaders
from claq.precomputed import PrecomputedDataset, PrecomputedSplit, save_precomputed_dataset
from claq.sensitive_labels import build_cifar10_sensitive_match, compute_s_from_image_features
from claq.utils import resolve_device, sha256_file


@torch.no_grad()
def collect_split(
    loader,
    *,
    model_clip,
    dictionary,
    answering_model,
    sensitive_indices,
    device,
    response_threshold: float,
    sensitive_tau: float,
    sensitive_topk: int,
) -> PrecomputedSplit:
    responses_all = []
    labels_all = []
    sensitive_all = []
    for images, labels in tqdm(loader, leave=False):
        image_features = encode_images(model_clip=model_clip, images=images, device=device)
        responses = concept_answers_from_image_features(
            image_features=image_features,
            dictionary=dictionary,
            answering_model=answering_model,
            train_device=device,
            threshold=response_threshold,
        )
        _, sensitive = compute_s_from_image_features(
            image_features=image_features,
            logit_scale=model_clip.logit_scale.exp(),
            dictionary=dictionary,
            sens_idx=sensitive_indices,
            tau=sensitive_tau,
            topk=sensitive_topk,
        )
        responses_all.append(responses.cpu())
        labels_all.append(labels.to(torch.long).cpu())
        sensitive_all.append(sensitive.to(torch.long).cpu())
    return PrecomputedSplit(
        responses=torch.cat(responses_all),
        labels=torch.cat(labels_all),
        sensitive=torch.cat(sensitive_all),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--concepts", required=True, type=Path)
    parser.add_argument("--concept-qa-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--clip-model", default="ViT-B/16")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--response-threshold", type=float, default=0.0)
    parser.add_argument("--sensitive-tau", type=float, default=0.7)
    parser.add_argument("--sensitive-topk", type=int, default=3)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    if args.sensitive_topk < 1:
        raise ValueError("--sensitive-topk must be at least 1")
    if not torch.isfinite(torch.tensor(args.response_threshold)):
        raise ValueError("--response-threshold must be finite")
    if not torch.isfinite(torch.tensor(args.sensitive_tau)):
        raise ValueError("--sensitive-tau must be finite")
    device = resolve_device(args.device)
    model_clip, preprocess = load_clip_model(args.clip_model, device=device)
    concepts = load_concepts(args.concepts)
    dictionary = build_concept_dictionary(model_clip, concepts, device)
    answering_model = load_concept_qa_checkpoint(args.concept_qa_checkpoint, device)
    sensitive_match = build_cifar10_sensitive_match(concepts)
    if sensitive_match.missing:
        raise ValueError(f"Missing designated concepts: {sensitive_match.missing}")
    if args.sensitive_topk > int(sensitive_match.indices.numel()):
        raise ValueError("--sensitive-topk exceeds the number of designated sensitive concepts")
    train_loader, validation_loader, test_loader = get_cifar10_train_valid_test_loaders(
        transform=preprocess,
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        shuffle_train=False,
        download=args.download,
    )
    common = dict(
        model_clip=model_clip,
        dictionary=dictionary,
        answering_model=answering_model,
        sensitive_indices=sensitive_match.indices,
        device=device,
        response_threshold=args.response_threshold,
        sensitive_tau=args.sensitive_tau,
        sensitive_topk=args.sensitive_topk,
    )
    train = collect_split(train_loader, **common)
    validation = collect_split(validation_loader, **common)
    test = collect_split(test_loader, **common)
    query_set_mask = torch.zeros(len(concepts), dtype=torch.bool)
    query_set_mask[sensitive_match.indices] = True
    dataset = PrecomputedDataset(
        train=train,
        validation=validation,
        test=test,
        query_names=tuple(concepts),
        query_set_mask=query_set_mask,
        query_costs=torch.ones(len(concepts)),
        admissible_mask=torch.ones(len(concepts), dtype=torch.bool),
        justification_mask=torch.zeros(len(concepts), dtype=torch.bool),
        metadata={
            "dataset": "CIFAR-10",
            "target": "object class",
            "sensitive_variable": (
                "binary CLIP-derived indicator: mean of the top-k normalized similarities "
                "within the designated concept set is at least tau"
            ),
            "sensitive_tau": args.sensitive_tau,
            "sensitive_topk": args.sensitive_topk,
            "designated_concepts": sensitive_match.matched,
            "response_source": "hard ConceptNet2 responses",
            "response_threshold": args.response_threshold,
            "clip_model": args.clip_model,
            "concept_qa_checkpoint": str(args.concept_qa_checkpoint),
            "concept_qa_checkpoint_sha256": sha256_file(args.concept_qa_checkpoint),
            "concept_specification": str(args.concepts),
            "concept_specification_sha256": sha256_file(args.concepts),
            "validation_fraction": args.validation_fraction,
            "split_seed": args.split_seed,
            "operational_response_alphabet": [-1, 1],
            "report_equalized_odds": False,
            "split_sizes": {
                "train": int(train.labels.numel()),
                "validation": int(validation.labels.numel()),
                "test": int(test.labels.numel()),
            },
        },
    )
    print(save_precomputed_dataset(args.output, dataset))


if __name__ == "__main__":
    main()
