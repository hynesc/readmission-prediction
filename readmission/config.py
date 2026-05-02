from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    raw_zip: str = "resource/S1_Data.zip"
    raw_member: str = "S1_Data.txt"
    output_dir: str = "artifacts/strict"
    seed: int = 13
    valid_fraction: float = 0.15
    test_fraction: float = 0.20
    max_visits: int | None = None
    embedding_dim: int = 100
    word2vec_epochs: int = 10
    train_word2vec: bool = False


@dataclass
class ModelConfig:
    model_type: str = "gru"
    embedding_type: str = "code_pool"
    word2vec_path: str | None = None
    freeze_embeddings: bool = False
    code_embedding_dim: int = 128
    visit_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.1
    num_topics: int = 50
    kl_weight: float = 1.0
    transformer_heads: int = 4
    transformer_ff_dim: int = 256


@dataclass
class TrainConfig:
    dataset_dir: str = "artifacts/strict"
    output_dir: str = "runs/gru"
    seed: int = 13
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    device: str = "auto"
    num_workers: int = 0


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _merge_dataclass(cls: type, values: dict[str, Any] | None):
    base = cls()
    if not values:
        return base
    allowed = set(base.__dataclass_fields__)  # type: ignore[attr-defined]
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**{**base.__dict__, **values})


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return ExperimentConfig(
        data=_merge_dataclass(DataConfig, raw.get("data")),
        model=_merge_dataclass(ModelConfig, raw.get("model")),
        train=_merge_dataclass(TrainConfig, raw.get("train")),
    )


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data": config.data.__dict__,
        "model": config.model.__dict__,
        "train": config.train.__dict__,
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=True)
