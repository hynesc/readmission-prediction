# Revisiting Neural Models for Hospital Readmission

**TL;DR:** We replicated the CONTENT readmission model in PyTorch and benchmarked tuned CONTENT vs GRU. The repository now also includes a script-based strict-forward experiment pipeline that fixes the main post-hoc audit issues before comparing GRU, CONTENT, and a causal Transformer.

**Links:** [Original Paper (PLOS ONE)](https://doi.org/10.1371/journal.pone.0195024) | [Notebook (`code/CONTENT_colab.ipynb`)](./code/CONTENT_colab.ipynb) | [Audit Notes (`docs/audit.md`)](./docs/audit.md)

## Project Summary

During the group project, we:

1. Replicated the Theano-era CONTENT model and GRU baseline in PyTorch.
2. Re-ran experiments across multiple trials with modern tooling.
3. Added a tuning workflow to test whether CONTENT gains were configuration-sensitive.

After the group project concluded, I independently performed a post-hoc code audit to document bugs and methodological risks.
I revisited the project because I wanted to understand why our implementation performed substantially better than the original report.

## Results Snapshot

| Model | PR-AUC | ROC-AUC |
| :-- | --: | --: |
| CONTENT (Original Paper) | 0.480 | 0.700 |
| CONTENT (This Replication - Tuned) | 0.646 | 0.801 |
| GRU Baseline | 0.632 | 0.791 |

*Note: This table is a compact portfolio summary. Some rows come from different experimental protocols. See [docs/audit.md](./docs/audit.md) for comparability caveats and implementation details. For full trial-level reporting (mean, standard deviation, and significance testing), see the project reports in this repository.*

## What We Improved

- Reimplemented CONTENT and GRU in PyTorch for maintainability and reproducibility.
- Added cleaner training/evaluation loops and repeated-trial metric reporting.
- Added a grid-search workflow for key hyperparameters (hidden size, learning rate, number of topics, epochs).
- Preserved end-to-end preprocessing and artifact saving for model outputs and latent representations.

## Quickstart

### Option A: Strict-forward scripts

Build corrected train/validation/test artifacts from the raw synthetic dataset:

```bash
python -m readmission.data build --config configs/strict.yaml
```

Train model variants under the same evaluation protocol:

```bash
python -m readmission.train --config configs/gru.yaml
python -m readmission.train --config configs/content_original.yaml
python -m readmission.train --config configs/content.yaml
python -m readmission.train --config configs/transformer.yaml
```

Each run writes `history.csv`, `metrics.json`, `config.yaml`, and `model.pt` under `runs/`.

### Option B: Notebook report

```bash
cd code
conda env create -f content-env.yaml
conda activate content-env
jupyter lab CONTENT_colab.ipynb
```

The notebook is retained as a report/runner for the original replication workflow. New experiments should use the package entry points above so preprocessing, labels, masking, and metric selection are reproducible.

## Data and Methods

- Dataset: synthetic CHF EHR dataset released with the original paper (`resource/S1_Data.zip`).
- Cohort size: 3,000 synthetic patients.
- Task: visit-level binary prediction for strict prospective 30-day readmission.
- Label policy: `current_day < future_inpatient_day <= current_day + 30`, excluding same-day inpatient events.
- Models: GRU baseline, original-style CONTENT, deterministic CONTENT-style, and causal Transformer.
- Primary metric: PR-AUC. Secondary metrics: ROC-AUC, F1, precision, recall, and accuracy.
- Thresholded metrics use a threshold selected on validation F1, then applied to test.
- Preprocessing splits patients first, fits vocabulary and optional Word2Vec only on train, reserves `PAD=0` and `UNK=1`, aggregates by `PID, DAY_ID`, and sorts visits chronologically.

### CONTENT variants

This repository keeps two CONTENT-family models because they answer different questions:

- `content_original` (`configs/content_original.yaml`): original-style reproduction. It uses multi-hot visit inputs, a GRU branch, a patient-level variational topic posterior with `mu` and `log_sigma`, reparameterized `theta`, an additive topic logit contribution, and a KL regularization term. Padding is still handled with explicit masks, so it is faithful in architecture but corrected in implementation details.
- `content` (`configs/content.yaml`): corrected deterministic CONTENT-style model. It keeps the GRU plus topic-context idea, but replaces the variational topic layer with a masked deterministic topic summary. This is useful as a cleaner ablation, but it should not be described as the original CONTENT architecture.

## Post-Hoc Audit (Important)

This section reflects my independent post-project review. I identified several issues:

- a critical padding/loss bug in one original training path,
- an evaluation bug in an original fixed-batch script,
- a transductive embedding contamination risk in the replication tooling,
- and shared task-design caveats around label definition and temporal causality.

At a high level, our stronger performance was mainly explained by cleaner loss handling around padding, stronger initialization/tuning choices, and implementation differences from legacy code paths that contained critical bugs.

For technical details and file-level notes, see [docs/audit.md](./docs/audit.md).

<details>
<summary>One-paragraph interpretation</summary>
Our group-project conclusion is that CONTENT can outperform baseline under some settings, but those gains are sensitive to implementation details. My follow-up audit is included so readers can separate likely architecture effects from pipeline artifacts.
</details>

## Repository Layout

- `readmission/`: script-based strict-forward preprocessing, modeling, training, and metrics.
- `configs/`: reproducible YAML configs for corrected data build and model runs.
- `tests/`: focused unit tests for preprocessing, metrics, config loading, masking, CONTENT padding behavior, original-style CONTENT KL wiring, and Transformer causal masks.
- `code/CONTENT_colab.ipynb`: legacy notebook report/runner.
- `code/get_embedding.py`: local script used to generate Word2Vec embeddings.
- `resource/`: dataset, preprocessed splits, vocabulary, embeddings.
- `output/`: saved predictions, hidden states, and topic vectors.
- `docs/audit.md`: post-hoc bug and risk analysis.
