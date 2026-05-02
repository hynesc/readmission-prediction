from __future__ import annotations

from readmission.config import ModelConfig


def build_model(config: ModelConfig, vocab: dict[str, int]):
    vocab_size = len(vocab)
    if config.model_type == "gru":
        from .gru import GRUReadmissionModel

        return GRUReadmissionModel(vocab, config)
    if config.model_type == "content":
        from .content import CONTENTModel

        return CONTENTModel(vocab, config)
    if config.model_type == "content_original":
        from .content_original import OriginalStyleCONTENTModel

        return OriginalStyleCONTENTModel(vocab, config)
    if config.model_type == "transformer":
        from .transformer import CausalTransformerReadmissionModel

        return CausalTransformerReadmissionModel(vocab, config)
    raise ValueError(f"Unknown model_type: {config.model_type}")
