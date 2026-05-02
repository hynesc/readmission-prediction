from __future__ import annotations

import math

from readmission.config import ModelConfig

from .visit import nn, torch, build_visit_encoder


class CausalTransformerReadmissionModel(nn.Module):
    def __init__(self, vocab: dict[str, int], config: ModelConfig):
        super().__init__()
        self.visit_encoder = build_visit_encoder(vocab, config)
        layer = nn.TransformerEncoderLayer(
            d_model=config.visit_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.output = nn.Linear(config.visit_dim, 1)

    @staticmethod
    def causal_mask(length: int, device):
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    @staticmethod
    def sinusoidal_positions(length: int, dim: int, device):
        positions = torch.arange(length, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
        pe = torch.zeros(length, dim, device=device)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term[: pe[:, 1::2].shape[1]])
        return pe

    def forward(self, codes, code_mask, visit_mask):
        visits = self.visit_encoder(codes, code_mask)
        visits = visits + self.sinusoidal_positions(visits.size(1), visits.size(2), visits.device).unsqueeze(0)
        hidden = self.encoder(
            visits,
            mask=self.causal_mask(visits.size(1), visits.device),
            src_key_padding_mask=~visit_mask.bool(),
        )
        return self.output(hidden).squeeze(-1)
