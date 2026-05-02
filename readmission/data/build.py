from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from readmission.config import DataConfig, load_config

from .preprocess import aggregate_patient_sequences, fit_vocab, read_raw_rows, split_patient_ids, train_visit_corpus


def _sequence_to_json(patient) -> dict:
    return {
        "pid": patient.pid,
        "days": list(patient.days),
        "visits": [list(visit.codes) for visit in patient.visits],
        "labels": list(patient.labels),
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def _maybe_train_word2vec(corpus: list[list[str]], output_path: Path, dim: int, epochs: int, seed: int) -> None:
    try:
        from gensim.models import Word2Vec
    except ImportError as exc:
        raise RuntimeError("gensim is required for train_word2vec=true") from exc
    model = Word2Vec(
        corpus,
        vector_size=dim,
        window=5,
        min_count=1,
        workers=1,
        seed=seed,
        sg=1,
    )
    model.train(corpus, total_examples=len(corpus), epochs=epochs)
    model.wv.save_word2vec_format(str(output_path), binary=False)


def build_dataset(config: DataConfig) -> dict[str, object]:
    rows = read_raw_rows(config.raw_zip, config.raw_member)
    splits = split_patient_ids((row["PID"] for row in rows), config.valid_fraction, config.test_fraction, config.seed)
    train_ids = set(splits["train"])
    vocab = fit_vocab(rows, train_ids)
    index_to_token = {idx: token for token, idx in vocab.items()}

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_counts = {}
    train_sequences = []
    for split, patient_ids in splits.items():
        sequences = aggregate_patient_sequences(rows, patient_ids, vocab, max_visits=config.max_visits)
        if split == "train":
            train_sequences = sequences
        split_counts[split] = len(sequences)
        _write_jsonl(output_dir / f"{split}.jsonl", [_sequence_to_json(patient) for patient in sequences])
        with open(output_dir / f"{split}.pkl", "wb") as f:
            pickle.dump(sequences, f)

    with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, sort_keys=True)
    with open(output_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, sort_keys=True)

    if config.train_word2vec:
        corpus = train_visit_corpus(train_sequences, index_to_token)
        _maybe_train_word2vec(corpus, output_dir / "word2vec.vector", config.embedding_dim, config.word2vec_epochs, config.seed)

    manifest = {
        "raw_zip": config.raw_zip,
        "split_counts": split_counts,
        "vocab_size": len(vocab),
        "pad_index": 0,
        "unk_index": 1,
        "label_policy": "strict_forward_30d: current_day < inpatient_day <= current_day + 30",
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        manifest = build_dataset(load_config(args.config).data)
        print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
