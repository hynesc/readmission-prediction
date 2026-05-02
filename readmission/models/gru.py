from __future__ import annotations

from readmission.config import ModelConfig

from .visit import nn, build_visit_encoder


class GRUReadmissionModel(nn.Module):
    def __init__(self, vocab: dict[str, int], config: ModelConfig):
        super().__init__()
        self.visit_encoder = build_visit_encoder(vocab, config)
        self.gru = nn.GRU(
            config.visit_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(config.hidden_dim, 1)

    def forward(self, codes, code_mask, visit_mask):
        visits = self.visit_encoder(codes, code_mask)
        lengths = visit_mask.sum(dim=1).cpu().clamp_min(1)
        packed = nn.utils.rnn.pack_padded_sequence(visits, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=visits.size(1))
        return self.output(hidden).squeeze(-1)
