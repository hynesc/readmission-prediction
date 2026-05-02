from __future__ import annotations

import csv
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .schema import PAD_INDEX, PAD_TOKEN, UNK_INDEX, UNK_TOKEN, PatientSequence, Visit

INPATIENT_LOCATION = "INPATIENT HOSPITAL"


RawRow = dict[str, str]


def read_raw_rows(zip_path: str | Path, member: str = "S1_Data.txt") -> list[RawRow]:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as f:
            text = (line.decode("utf-8") for line in f)
            return list(csv.DictReader(text, delimiter="\t"))


def split_patient_ids(
    patient_ids: Iterable[str],
    valid_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[str]]:
    ids = sorted({str(pid) for pid in patient_ids})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_test = int(round(n * test_fraction))
    n_valid = int(round(n * valid_fraction))
    test = sorted(ids[:n_test])
    valid = sorted(ids[n_test : n_test + n_valid])
    train = sorted(ids[n_test + n_valid :])
    if not train or not valid or not test:
        raise ValueError("Split fractions produced an empty train, validation, or test split")
    return {"train": train, "valid": valid, "test": test}


def fit_vocab(rows: Iterable[RawRow], train_ids: set[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if row["PID"] in train_ids:
            token = row["DX_GROUP_DESCRIPTION"].strip()
            if token:
                counts[token] += 1
    vocab = {PAD_TOKEN: PAD_INDEX, UNK_TOKEN: UNK_INDEX}
    for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        vocab[token] = len(vocab)
    return vocab


def aggregate_patient_sequences(
    rows: Iterable[RawRow],
    patient_ids: Iterable[str],
    vocab: dict[str, int],
    max_visits: int | None = None,
) -> list[PatientSequence]:
    wanted = set(patient_ids)
    grouped: dict[str, dict[int, dict[str, set[str]]]] = defaultdict(
        lambda: defaultdict(lambda: {"codes": set(), "locations": set()})
    )
    for row in rows:
        pid = row["PID"]
        if pid not in wanted:
            continue
        day = int(row["DAY_ID"])
        code = vocab.get(row["DX_GROUP_DESCRIPTION"].strip(), UNK_INDEX)
        location = row["SERVICE_LOCATION"].strip()
        grouped[pid][day]["codes"].add(code)
        if location:
            grouped[pid][day]["locations"].add(location)

    sequences: list[PatientSequence] = []
    for pid in sorted(grouped):
        visits = []
        inpatient_days = sorted(
            day
            for day, payload in grouped[pid].items()
            if INPATIENT_LOCATION in payload["locations"]
        )
        for day in sorted(grouped[pid]):
            payload = grouped[pid][day]
            visits.append(
                Visit(
                    day=day,
                    codes=tuple(sorted(payload["codes"])),
                    service_locations=tuple(sorted(payload["locations"])),
                )
            )
        if max_visits is not None:
            visits = visits[:max_visits]
        labels = tuple(strict_forward_labels([visit.day for visit in visits], inpatient_days))
        sequences.append(PatientSequence(pid=pid, visits=tuple(visits), labels=labels))
    return sequences


def strict_forward_labels(days: Iterable[int], inpatient_days: Iterable[int], window: int = 30) -> list[int]:
    inpatient = sorted(set(inpatient_days))
    labels: list[int] = []
    for day in days:
        labels.append(int(any(day < future_day <= day + window for future_day in inpatient)))
    return labels


def train_visit_corpus(sequences: Iterable[PatientSequence], index_to_token: dict[int, str]) -> list[list[str]]:
    corpus: list[list[str]] = []
    for patient in sequences:
        for visit in patient.visits:
            tokens = [index_to_token[idx] for idx in visit.codes if idx > UNK_INDEX]
            if tokens:
                corpus.append(tokens)
    return corpus
