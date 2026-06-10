from __future__ import annotations

import torch.nn as nn


class ConceptAnswererMLP(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dims: tuple[int, ...] = (512, 256, 128, 64),
        dropout: float = 0.0,
        use_batch_norm: bool = True,
        activation: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_dim = embed_dim * 2
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        layers: list[nn.Module] = []
        in_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class ConceptNet2(nn.Module):
    def __init__(self, embed_dims: int = 512):
        super().__init__()
        self.embed_dims = embed_dims
        self.input_dim = self.embed_dims * 2

        self.layer1 = nn.Linear(self.input_dim, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 64)

        self.norm1 = nn.BatchNorm1d(512)
        self.norm2 = nn.BatchNorm1d(256)
        self.norm3 = nn.BatchNorm1d(128)
        self.norm4 = nn.BatchNorm1d(64)

        self.relu = nn.ReLU()
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.norm1(self.layer1(x)))
        x = self.relu(self.norm2(self.layer2(x)))
        x = self.relu(self.norm3(self.layer3(x)))
        x = self.relu(self.norm4(self.layer4(x)))
        return self.head(x).squeeze(-1)
