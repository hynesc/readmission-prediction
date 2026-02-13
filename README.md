# Revisiting Neural Models for Hospital Readmission

**TL;DR:** We replicated the CONTENT readmission model in PyTorch and benchmarked tuned CONTENT vs GRU. After the group project, I conducted a post-hoc audit of implementation bugs and methodological risks.

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

### Option A: Run in Colab (recommended)

1. Upload this repository to Google Drive.
2. Open `code/CONTENT_colab.ipynb` in Colab.
3. Select GPU runtime (`Runtime` -> `Change runtime type` -> `T4 GPU`).
4. Run all cells.

### Option B: Local environment

```bash
cd code
conda env create -f content-env.yaml
conda activate content-env
jupyter lab CONTENT_colab.ipynb
```

## Data and Methods

- Dataset: synthetic CHF EHR dataset released with the original paper (`resource/S1_Data.zip`).
- Cohort size: 3,000 synthetic patients.
- Task: visit-level binary prediction for 30-day readmission.
- Models: CONTENT (GRU + topic branch) and GRU baseline.
- Primary metrics: PR-AUC and ROC-AUC.

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

- `code/CONTENT_colab.ipynb`: main end-to-end notebook (preprocess, train, evaluate, tune).
- `code/get_embedding.py`: local script used to generate Word2Vec embeddings.
- `resource/`: dataset, preprocessed splits, vocabulary, embeddings.
- `output/`: saved predictions, hidden states, and topic vectors.
- `docs/audit.md`: post-hoc bug and risk analysis.
