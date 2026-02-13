# Audit Notes: Replication and Original-Code Risks

This document summarizes issues I identified after our initial replication write-up.
It is intended to improve transparency and reproducibility for future readers.

## Scope

I reviewed:

- the original Theano code release (not stored in this repo; key files: `CONTENT.py`, `CONTENT_fixedBatchSize.py`),
- our replication notebook at `code/CONTENT_colab.ipynb`,
- and embedding generation tooling at `code/get_embedding.py`.

## Key Findings

### 1) Critical padding/loss bug in original training path

In the original `CONTENT.py`, padded labels are initialized to ones and not excluded from the flattened BCE objective.  
At the same time, model outputs for padded positions are mask-suppressed near zero.

Impact:

- training receives a large, incorrect gradient contribution from padding,
- signal-to-noise for real visits is reduced,
- and performance can be significantly depressed.

### 2) Evaluation bug in original fixed-batch script

In the original `CONTENT_fixedBatchSize.py`, one test-evaluation path appends input-derived values instead of model predictions.

Impact:

- reported metrics from that path are not reliable.

### 3) Embedding contamination risk in replication tooling

In `code/get_embedding.py`, Word2Vec is fit using all splits (`X_train.pkl`, `X_valid.pkl`, `X_test.pkl`) and later used to initialize CONTENT embeddings.

Impact:

- this is transductive contamination (test distribution seen during representation learning),
- and it can inflate downstream test performance relative to strict train-only protocols.

### 4) Shared label-definition caveat

Both pipelines use readmission tagging logic that includes same-day inpatient events in the window (`day <= DAY_ID < day + 30`).

Impact:

- depending on intended clinical framing, this can make the task easier than strictly forward-looking prediction.

### 5) Shared temporal-causality caveat

Both implementations use global patient context in a way that can include information from future visits when producing per-visit outputs.

Impact:

- this is not strictly causal at time `t`,
- and it may overstate prospective deployment performance.

## Recommendations from This Review

1. Fit embeddings on train split only.
2. Enforce causal context for per-visit prediction.
3. Revisit labeling to exclude index-event self-inclusion if required by deployment semantics.
4. Keep masked loss and masked metric extraction only over valid visits.
5. Report both replication-compatible and strict-causal results side by side.
