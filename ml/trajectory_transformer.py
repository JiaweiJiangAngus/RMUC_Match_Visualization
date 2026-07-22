#!/usr/bin/env python3
"""Temporal Transformer used by RMUC multi-agent trajectory training."""

from __future__ import annotations

import torch
from torch import nn


HISTORY_TOKEN_COUNT = 4
HISTORY_TOKEN_WIDTH = 54
HISTORY_FEATURE_DIM = HISTORY_TOKEN_COUNT * HISTORY_TOKEN_WIDTH


class TemporalBattlefieldTransformer(nn.Module):
    """Encode four battlefield snapshots and one target-context token.

    This is an actual Transformer encoder: every token is normalized and fed
    through learned multi-head self-attention and feed-forward blocks.  The
    final target-context token attends to all four temporal battlefield tokens
    before producing multi-horizon coordinate residuals.
    """

    def __init__(
        self,
        input_dim: int,
        horizon_count: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if input_dim <= HISTORY_FEATURE_DIM:
            raise ValueError("input_dim does not contain a target-context suffix")
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.input_dim = input_dim
        self.horizon_count = horizon_count
        self.context_dim = input_dim - HISTORY_FEATURE_DIM
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward

        self.history_projection = nn.Linear(HISTORY_TOKEN_WIDTH, d_model)
        self.context_projection = nn.Linear(self.context_dim, d_model)
        self.position_embedding = nn.Parameter(
            torch.empty(1, HISTORY_TOKEN_COUNT + 1, d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, horizon_count * 2)

        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        history = inputs[:, :HISTORY_FEATURE_DIM].reshape(
            -1, HISTORY_TOKEN_COUNT, HISTORY_TOKEN_WIDTH
        )
        context = inputs[:, HISTORY_FEATURE_DIM:]
        tokens = torch.cat(
            (
                self.history_projection(history),
                self.context_projection(context).unsqueeze(1),
            ),
            dim=1,
        )
        encoded = self.encoder(tokens + self.position_embedding)
        target_token = self.norm(encoded[:, -1])
        return self.head(target_token).reshape(-1, self.horizon_count, 2)

