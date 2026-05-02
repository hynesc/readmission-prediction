from __future__ import annotations


def require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for model training") from exc
    return torch, nn


torch, nn = require_torch()


def load_word2vec_matrix(vocab: dict[str, int], word2vec_path: str):
    vectors: dict[str, list[float]] = {}
    with open(word2vec_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        if len(header) != 2:
            raise ValueError(f"Invalid word2vec header in {word2vec_path}")
        dim = int(header[1])
        for line in f:
            parts = line.rstrip().split()
            if len(parts) < dim + 1:
                continue
            token = " ".join(parts[:-dim])
            if token in vocab:
                vectors[token] = [float(value) for value in parts[-dim:]]
    matrix = torch.zeros(len(vocab), dim)
    for token, idx in vocab.items():
        if token in vectors:
            matrix[idx] = torch.tensor(vectors[token], dtype=torch.float32)
    return matrix


class CodePoolVisitEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        code_embedding_dim: int,
        visit_dim: int,
        pad_index: int = 0,
        pretrained=None,
        freeze: bool = False,
    ):
        super().__init__()
        if pretrained is None:
            self.embedding = nn.Embedding(vocab_size, code_embedding_dim, padding_idx=pad_index)
        else:
            self.embedding = nn.Embedding.from_pretrained(pretrained, freeze=freeze, padding_idx=pad_index)
            code_embedding_dim = pretrained.size(1)
        self.projection = nn.Linear(code_embedding_dim, visit_dim)

    def forward(self, codes, code_mask):
        embedded = self.embedding(codes)
        mask = code_mask.unsqueeze(-1).float()
        summed = (embedded * mask).sum(dim=2)
        denom = mask.sum(dim=2).clamp_min(1.0)
        pooled = summed / denom
        return self.projection(pooled)


class MultiHotVisitEncoder(nn.Module):
    def __init__(self, vocab_size: int, visit_dim: int):
        super().__init__()
        self.projection = nn.Linear(vocab_size, visit_dim)

    def forward(self, codes, code_mask):
        visits = torch.zeros(codes.size(0), codes.size(1), self.projection.in_features, device=codes.device)
        visits.scatter_add_(2, codes.clamp_min(0), code_mask.float())
        visits[:, :, 0] = 0.0
        visits[:, :, 1] = 0.0
        return self.projection(visits)


def build_visit_encoder(vocab: dict[str, int], config):
    if config.embedding_type == "multi_hot":
        return MultiHotVisitEncoder(len(vocab), config.visit_dim)
    if config.embedding_type in {"code_pool", "word2vec"}:
        pretrained = None
        if config.embedding_type == "word2vec":
            if not config.word2vec_path:
                raise ValueError("model.word2vec_path is required when embedding_type=word2vec")
            pretrained = load_word2vec_matrix(vocab, config.word2vec_path)
        return CodePoolVisitEncoder(
            len(vocab),
            config.code_embedding_dim,
            config.visit_dim,
            pretrained=pretrained,
            freeze=config.freeze_embeddings,
        )
    raise ValueError(f"Unknown embedding_type: {config.embedding_type}")
