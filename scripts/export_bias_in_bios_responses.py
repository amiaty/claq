#!/usr/bin/env python3
"""Export Bias in Bios Concept-QA responses to the canonical CLAQ .npz format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from claq.models import ConceptAnswererMLP
from claq.precomputed import PrecomputedDataset, PrecomputedSplit, save_precomputed_dataset
from claq.utils import resolve_device, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="Directory containing train.csv, validation.csv, and test.csv")
    parser.add_argument("--concepts", required=True, type=Path,
                        help="bias_in_bios.csv concept specification")
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="ConceptAnswererMLP checkpoint")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--text-column", default="hard_text")
    parser.add_argument("--target-column", default="profession")
    parser.add_argument("--sensitive-column", default="gender")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    device = resolve_device(args.device)
    concepts = pd.read_csv(args.concepts)
    required = {"concept", "kind", "description"}
    if not required.issubset(concepts.columns):
        raise ValueError(f"Concept file must contain columns {sorted(required)}")
    concept_names = concepts["concept"].astype(str).str.strip().tolist()
    if any(not name for name in concept_names):
        raise ValueError("Concept names must be nonempty")
    if len(set(concept_names)) != len(concept_names):
        raise ValueError("Concept names must be unique")
    query_set_mask = concepts["kind"].astype(str).ne("utility").to_numpy(dtype=bool)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = dict(checkpoint["model_config"])
    model_config["hidden_dims"] = tuple(model_config["hidden_dims"])
    model = ConceptAnswererMLP(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    encoder_name = checkpoint["encoder_name"]
    encoder = SentenceTransformer(encoder_name, device=str(device))
    concept_embeddings = checkpoint["concept_embeddings"].to(device).to(torch.float32)
    if concept_embeddings.shape[0] != len(concept_names):
        raise ValueError("Checkpoint and concept file have different query counts")
    threshold = float(checkpoint.get("decision_threshold", 0.5))
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Concept-QA decision threshold must lie in [0,1]")

    split_objects: dict[str, PrecomputedSplit] = {}
    with torch.no_grad():
        for split in ("train", "validation", "test"):
            frame = pd.read_csv(args.data_dir / f"{split}.csv")
            for column in (args.text_column, args.target_column, args.sensitive_column):
                if column not in frame.columns:
                    raise KeyError(f"{split}.csv is missing '{column}'")
            text_embeddings = encoder.encode(
                frame[args.text_column].fillna("").astype(str).tolist(),
                batch_size=args.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            ).astype(np.float32)
            text_tensor = torch.from_numpy(text_embeddings).to(device)
            score_chunks = []
            for start in range(0, len(text_tensor), args.batch_size):
                text_batch = text_tensor[start:start + args.batch_size]
                repeated_text = text_batch.repeat_interleave(len(concept_names), dim=0)
                repeated_concepts = concept_embeddings.repeat(text_batch.size(0), 1)
                inputs = torch.cat([repeated_text, repeated_concepts], dim=1)
                logits = model(inputs).view(text_batch.size(0), len(concept_names))
                score_chunks.append(torch.sigmoid(logits).cpu())
            probabilities = torch.cat(score_chunks, dim=0)
            responses = torch.where(
                probabilities >= threshold,
                torch.ones_like(probabilities),
                -torch.ones_like(probabilities),
            )
            label_values = pd.to_numeric(frame[args.target_column], errors="raise").to_numpy(dtype=float)
            sensitive_values = pd.to_numeric(frame[args.sensitive_column], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(label_values).all() or not np.isfinite(sensitive_values).all():
                raise ValueError(f"{split}.csv contains non-finite target or sensitive values")
            if not np.array_equal(label_values, np.rint(label_values)):
                raise ValueError(f"{split}.csv target labels must be integer-valued")
            if not np.array_equal(sensitive_values, np.rint(sensitive_values)):
                raise ValueError(f"{split}.csv sensitive labels must be integer-valued")
            split_objects[split] = PrecomputedSplit(
                responses=responses,
                labels=torch.as_tensor(np.rint(label_values), dtype=torch.long),
                sensitive=torch.as_tensor(np.rint(sensitive_values), dtype=torch.long),
            )

    dataset = PrecomputedDataset(
        train=split_objects["train"],
        validation=split_objects["validation"],
        test=split_objects["test"],
        query_names=tuple(concept_names),
        query_set_mask=torch.as_tensor(query_set_mask, dtype=torch.bool),
        query_costs=torch.ones(len(concept_names), dtype=torch.float32),
        admissible_mask=torch.ones(len(concept_names), dtype=torch.bool),
        justification_mask=torch.zeros(len(concept_names), dtype=torch.bool),
        metadata={
            "dataset": "Bias in Bios",
            "target": args.target_column,
            "sensitive": args.sensitive_column,
            "response_source": "ConceptAnswererMLP hard responses",
            "encoder_name": encoder_name,
            "decision_threshold": threshold,
            "concept_checkpoint": str(args.checkpoint),
            "concept_checkpoint_sha256": sha256_file(args.checkpoint),
            "concept_specification": str(args.concepts),
            "concept_specification_sha256": sha256_file(args.concepts),
            "operational_response_alphabet": [-1, 1],
            "report_equalized_odds": True,
            "split_sizes": {name: int(obj.labels.numel()) for name, obj in split_objects.items()},
        },
    )
    output = save_precomputed_dataset(args.output, dataset)
    print(output)


if __name__ == "__main__":
    main()
