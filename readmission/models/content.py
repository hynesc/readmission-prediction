from __future__ import annotations

from readmission.config import ModelConfig

from .visit import nn, torch, build_visit_encoder


class CONTENTModel(nn.Module):
    def __init__(self, vocab: dict[str, int], config: ModelConfig):
        super().__init__()
        self.visit_encoder = build_visit_encoder(vocab, config)
        self.gru = nn.GRU(config.visit_dim, config.hidden_dim, batch_first=True)
        self.topic_logits = nn.Linear(config.visit_dim, config.num_topics)
        self.topic_to_hidden = nn.Linear(config.num_topics, config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim * 2, 1)

    def topic_summary(self, visits, visit_mask):
        topic_logits = self.topic_logits(visits)
        topic_probs = torch.softmax(topic_logits, dim=-1)
        mask = visit_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (topic_probs * mask).sum(dim=1) / denom

    def forward(self, codes, code_mask, visit_mask):
        visits = self.visit_encoder(codes, code_mask)
        lengths = visit_mask.sum(dim=1).cpu().clamp_min(1)
        packed = nn.utils.rnn.pack_padded_sequence(visits, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=visits.size(1))
        topic_context = self.topic_to_hidden(self.topic_summary(visits, visit_mask))
        topic_context = topic_context.unsqueeze(1).expand(-1, hidden.size(1), -1)
        return self.output(torch.cat([hidden, topic_context], dim=-1)).squeeze(-1)
