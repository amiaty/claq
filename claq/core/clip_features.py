"""CLIP and concept-dictionary helpers."""

from __future__ import annotations

from pathlib import Path

import clip
import torch


def load_clip_model(model_name: str, device: torch.device):
    model, preprocess = clip.load(model_name, device=device)
    model = model.to(device).eval()
    return model, preprocess


def load_concepts(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle.readlines() if line.strip()]


@torch.no_grad()
def build_concept_dictionary(model_clip, concepts: list[str], device: torch.device) -> torch.Tensor:
    text = clip.tokenize(concepts).to(device)
    text_features = model_clip.encode_text(text)
    dictionary = text_features.T
    dictionary = dictionary / torch.linalg.norm(dictionary, dim=0, keepdim=True)
    return dictionary.to(device)


def encode_images(model_clip, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    feats = model_clip.encode_image(images.to(device))
    return feats / torch.linalg.norm(feats, dim=1, keepdim=True)


def compute_similarity_scores(image_features: torch.Tensor, dictionary: torch.Tensor, logit_scale, eps: float = 1e-12) -> torch.Tensor:
    sims = logit_scale * image_features @ dictionary
    maxv = sims.amax(dim=1, keepdim=True)
    minv = sims.amin(dim=1, keepdim=True)
    return (sims - minv) / (maxv - minv + eps)


def build_concept_qa_inputs(image_features: torch.Tensor, dictionary: torch.Tensor) -> torch.Tensor:
    batch_size = image_features.size(0)
    num_queries = dictionary.size(1)
    img_rep = image_features.unsqueeze(1).repeat(1, num_queries, 1)
    txt_rep = dictionary.T.unsqueeze(0).repeat(batch_size, 1, 1)
    return torch.cat([img_rep, txt_rep], dim=2).reshape(batch_size * num_queries, -1)


@torch.no_grad()
def concept_qa_batch_inputs(model_clip, images: torch.Tensor, dictionary: torch.Tensor, device: torch.device):
    image_features = encode_images(model_clip=model_clip, images=images, device=device)
    sims = compute_similarity_scores(
        image_features=image_features,
        dictionary=dictionary,
        logit_scale=model_clip.logit_scale.exp(),
    )
    inputs = build_concept_qa_inputs(image_features=image_features, dictionary=dictionary)
    return inputs.float(), sims
