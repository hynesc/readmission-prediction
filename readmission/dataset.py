from __future__ import annotations

import json
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_vocab_size(dataset_dir: str | Path) -> int:
    with open(Path(dataset_dir) / "vocab.json", "r", encoding="utf-8") as f:
        return len(json.load(f))


def load_vocab(dataset_dir: str | Path) -> dict[str, int]:
    with open(Path(dataset_dir) / "vocab.json", "r", encoding="utf-8") as f:
        return json.load(f)


def require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for training") from exc
    return torch, Dataset, DataLoader


def make_torch_dataset(records: list[dict]):
    torch, Dataset, _ = require_torch()

    class PatientDataset(Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, idx):
            return records[idx]

    return PatientDataset()


def collate_patients(batch: list[dict]):
    torch, _, _ = require_torch()
    batch_size = len(batch)
    max_visits = max(len(item["visits"]) for item in batch)
    max_codes = max(max((len(visit) for visit in item["visits"]), default=1) for item in batch)
    codes = torch.zeros(batch_size, max_visits, max_codes, dtype=torch.long)
    code_mask = torch.zeros(batch_size, max_visits, max_codes, dtype=torch.bool)
    visit_mask = torch.zeros(batch_size, max_visits, dtype=torch.bool)
    labels = torch.zeros(batch_size, max_visits, dtype=torch.float32)
    days = torch.zeros(batch_size, max_visits, dtype=torch.long)
    pids = []
    for i, item in enumerate(batch):
        pids.append(item["pid"])
        for j, visit_codes in enumerate(item["visits"]):
            visit_mask[i, j] = True
            labels[i, j] = float(item["labels"][j])
            days[i, j] = int(item["days"][j])
            if visit_codes:
                codes[i, j, : len(visit_codes)] = torch.tensor(visit_codes, dtype=torch.long)
                code_mask[i, j, : len(visit_codes)] = True
    return {"codes": codes, "code_mask": code_mask, "visit_mask": visit_mask, "labels": labels, "days": days, "pids": pids}


def make_loader(records: list[dict], batch_size: int, shuffle: bool, num_workers: int = 0):
    _, _, DataLoader = require_torch()
    return DataLoader(
        make_torch_dataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_patients,
    )
