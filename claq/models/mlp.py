"""Shared MLP used for the actor, classifier, and sensitive head."""

from __future__ import annotations

import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, query_size: int = 312, output_size: int = 312, eps: float | None = None, batchnorm: bool = False):
        super().__init__()
        self.query_size = query_size
        self.output_dim = output_size
        self.layer1 = nn.Linear(self.query_size, 2000)
        self.layer2 = nn.Linear(2000, 500)
        self.classifier = nn.Linear(500, self.output_dim)
        self.eps = eps

        if batchnorm:
            self.norm1 = nn.BatchNorm1d(2000)
            self.norm2 = nn.BatchNorm1d(500)
        else:
            self.norm1 = nn.LayerNorm(2000)
            self.norm2 = nn.LayerNorm(500)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.relu(self.norm1(self.layer1(x)))
        x = self.relu(self.norm2(self.layer2(x)))

        if self.eps is None:
            return self.classifier(x)

        if mask is None:
            raise ValueError("mask is required when the network is used as an actor")

        query_logits = self.classifier(x)
        query_mask = torch.where(mask == 1, -1e9, 0.0)
        query_logits = query_logits + query_mask.to(query_logits.device)

        query = self.softmax(query_logits / self.eps)
        # Straight-through hard selection with soft gradients.
        query = (self.softmax(query_logits / 1e-9) - query).detach() + query
        return query

    def change_eps(self, eps: float) -> None:
        self.eps = eps
