from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BinaryMetrics:
    pr_auc: float
    roc_auc: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    threshold: float


def _validate_binary_labels(labels: list[int]) -> None:
    if len(set(labels)) < 2:
        raise ValueError("Metrics require both positive and negative labels in the split")


def average_precision(labels: list[int], scores: list[float]) -> float:
    _validate_binary_labels(labels)
    paired = sorted(zip(scores, labels), reverse=True)
    positives = sum(labels)
    tp = 0
    total = 0
    area = 0.0
    for _, label in paired:
        total += 1
        if label:
            tp += 1
            area += tp / total
    return area / positives


def roc_auc(labels: list[int], scores: list[float]) -> float:
    _validate_binary_labels(labels)
    paired = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    rank_sum = 0.0
    i = 0
    while i < len(paired):
        j = i + 1
        while j < len(paired) and paired[j][0] == paired[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        rank_sum += avg_rank * sum(label for _, label in paired[i:j])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def threshold_metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, float]:
    preds = [int(score >= threshold) for score in scores]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "accuracy": accuracy}


def select_threshold(labels: list[int], scores: list[float]) -> float:
    _validate_binary_labels(labels)
    candidates = sorted(set(scores))
    best_threshold = candidates[0]
    best_f1 = -1.0
    for threshold in candidates:
        f1 = threshold_metrics(labels, scores, threshold)["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold


def compute_metrics(labels: list[int], scores: list[float], threshold: float) -> BinaryMetrics:
    _validate_binary_labels(labels)
    thresholded = threshold_metrics(labels, scores, threshold)
    return BinaryMetrics(
        pr_auc=average_precision(labels, scores),
        roc_auc=roc_auc(labels, scores),
        f1=thresholded["f1"],
        precision=thresholded["precision"],
        recall=thresholded["recall"],
        accuracy=thresholded["accuracy"],
        threshold=threshold,
    )
