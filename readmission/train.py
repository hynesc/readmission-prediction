from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path

from readmission.config import load_config, save_config
from readmission.dataset import load_jsonl, load_vocab, make_loader, require_torch
from readmission.metrics import compute_metrics, select_threshold
from readmission.models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch, _, _ = require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_bce_loss(logits, labels, visit_mask):
    torch, _, _ = require_torch()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    mask = visit_mask.float()
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def model_loss(model, logits, labels, visit_mask):
    loss = masked_bce_loss(logits, labels, visit_mask)
    if hasattr(model, "regularization_loss"):
        loss = loss + model.regularization_loss()
    return loss


def _device(name: str):
    torch, _, _ = require_torch()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _flatten_predictions(logits, labels, mask):
    probs = logits.sigmoid().detach().cpu()
    labels = labels.detach().cpu()
    mask = mask.detach().cpu().bool()
    return labels[mask].int().tolist(), probs[mask].float().tolist()


def evaluate(model, loader, device):
    model.eval()
    all_labels: list[int] = []
    all_scores: list[float] = []
    total_loss = 0.0
    total_batches = 0
    torch, _, _ = require_torch()
    with torch.no_grad():
        for batch in loader:
            codes = batch["codes"].to(device)
            code_mask = batch["code_mask"].to(device)
            visit_mask = batch["visit_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(codes, code_mask, visit_mask)
            loss = model_loss(model, logits, labels, visit_mask)
            y, p = _flatten_predictions(logits, labels, visit_mask)
            all_labels.extend(y)
            all_scores.extend(p)
            total_loss += float(loss.item())
            total_batches += 1
    return {"labels": all_labels, "scores": all_scores, "loss": total_loss / max(total_batches, 1)}


def run_training(config_path: str) -> dict[str, object]:
    config = load_config(config_path)
    set_seed(config.train.seed)
    device = _device(config.train.device)
    dataset_dir = Path(config.train.dataset_dir)
    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

    train_records = load_jsonl(dataset_dir / "train.jsonl")
    valid_records = load_jsonl(dataset_dir / "valid.jsonl")
    test_records = load_jsonl(dataset_dir / "test.jsonl")
    train_loader = make_loader(train_records, config.train.batch_size, shuffle=True, num_workers=config.train.num_workers)
    valid_loader = make_loader(valid_records, config.train.batch_size, shuffle=False, num_workers=config.train.num_workers)
    test_loader = make_loader(test_records, config.train.batch_size, shuffle=False, num_workers=config.train.num_workers)

    torch, _, _ = require_torch()
    model = build_model(config.model, load_vocab(dataset_dir)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)

    best = {"epoch": -1, "valid_pr_auc": -1.0, "state": None}
    history = []
    for epoch in range(1, config.train.epochs + 1):
        model.train()
        train_loss = 0.0
        batches = 0
        for batch in train_loader:
            codes = batch["codes"].to(device)
            code_mask = batch["code_mask"].to(device)
            visit_mask = batch["visit_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(codes, code_mask, visit_mask)
            loss = model_loss(model, logits, labels, visit_mask)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            batches += 1
        valid_eval = evaluate(model, valid_loader, device)
        threshold = select_threshold(valid_eval["labels"], valid_eval["scores"])
        valid_metrics = compute_metrics(valid_eval["labels"], valid_eval["scores"], threshold)
        row = {"epoch": epoch, "train_loss": train_loss / max(batches, 1), "valid_loss": valid_eval["loss"], **asdict(valid_metrics)}
        history.append(row)
        if valid_metrics.pr_auc > best["valid_pr_auc"]:
            best = {"epoch": epoch, "valid_pr_auc": valid_metrics.pr_auc, "state": copy.deepcopy(model.state_dict())}

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    valid_eval = evaluate(model, valid_loader, device)
    threshold = select_threshold(valid_eval["labels"], valid_eval["scores"])
    test_eval = evaluate(model, test_loader, device)
    valid_metrics = compute_metrics(valid_eval["labels"], valid_eval["scores"], threshold)
    test_metrics = compute_metrics(test_eval["labels"], test_eval["scores"], threshold)

    with open(output_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "best_epoch": best["epoch"],
        "valid": asdict(valid_metrics),
        "test": asdict(test_metrics),
        "model_type": config.model.model_type,
        "embedding_type": config.model.embedding_type,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run_training(args.config)


if __name__ == "__main__":
    main()
