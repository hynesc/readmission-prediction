from __future__ import annotations

from readmission.config import ModelConfig

from .visit import nn, torch, build_visit_encoder


class OriginalStyleCONTENTModel(nn.Module):
    """Original-style CONTENT with a variational patient topic branch.

    The 2017 Theano implementation used a patient-level topic posterior
    (`mu`, `log_sigma`), a sampled theta layer, and a KL term added to BCE.
    This PyTorch version keeps that structure while making padding explicit:
    topic summaries and KL normalization only use valid visits.
    """

    def __init__(self, vocab: dict[str, int], config: ModelConfig):
        super().__init__()
        self.kl_weight = config.kl_weight
        self.visit_encoder = build_visit_encoder(vocab, config)
        self.gru = nn.GRU(config.visit_dim, config.hidden_dim, batch_first=True)
        self.topic_visit = nn.Sequential(
            nn.Linear(config.visit_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(config.hidden_dim, config.num_topics)
        self.log_sigma = nn.Linear(config.hidden_dim, config.num_topics)
        self.visit_topic_logits = nn.Linear(config.visit_dim, config.num_topics)
        self.gru_logit = nn.Linear(config.hidden_dim, 1)
        self.last_kl_loss = None

    def posterior(self, visits, visit_mask):
        hidden = self.topic_visit(visits)
        mask = visit_mask.unsqueeze(-1).float()
        summary = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.mu(summary), self.log_sigma(summary)

    def sample_theta(self, mu, log_sigma):
        if self.training:
            epsilon = torch.randn_like(mu)
            return mu + epsilon * torch.exp(log_sigma)
        return mu

    @staticmethod
    def gaussian_kl(mu, log_sigma):
        variance = torch.exp(2.0 * log_sigma)
        return 0.5 * (mu.pow(2) + variance - 1.0 - 2.0 * log_sigma).sum(dim=1)

    def forward(self, codes, code_mask, visit_mask):
        visits = self.visit_encoder(codes, code_mask)
        lengths = visit_mask.sum(dim=1).cpu().clamp_min(1)
        packed = nn.utils.rnn.pack_padded_sequence(visits, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=visits.size(1))

        mu, log_sigma = self.posterior(visits, visit_mask)
        theta = self.sample_theta(mu, log_sigma)
        visit_topics = torch.softmax(self.visit_topic_logits(visits), dim=-1)
        theta_probs = torch.softmax(theta, dim=-1).unsqueeze(1)
        topic_logit = (visit_topics * theta_probs).sum(dim=-1)

        valid_patients = visit_mask.any(dim=1).float().sum().clamp_min(1.0)
        self.last_kl_loss = self.gaussian_kl(mu, log_sigma).sum() / valid_patients
        return self.gru_logit(hidden).squeeze(-1) + topic_logit

    def regularization_loss(self):
        if self.last_kl_loss is None:
            return 0.0
        return self.kl_weight * self.last_kl_loss
