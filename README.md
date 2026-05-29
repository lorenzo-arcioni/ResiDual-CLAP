# ResiDual for Audio: Spectral Reweighting of Residual Streams in CLAP Models

> **Status: Preliminary analysis in progress.**
> This repository documents an ongoing investigation into the internal representations of CLAP's audio encoder (HTS-AT), with the goal of designing spectral reweighting strategies — collectively referred to as **ResiDual** — to improve zero-shot downstream performance.

---

## Table of Contents

- [ResiDual for Audio: Spectral Reweighting of Residual Streams in CLAP Models](#residual-for-audio-spectral-reweighting-of-residual-streams-in-clap-models)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Structure](#project-structure)
    - [Top-level](#top-level)
    - [`src/`](#src)
      - [Configs](#configs)
      - [Datasets](#datasets)
      - [Models](#models)
      - [K-Folds](#k-folds)
      - [Notebooks](#notebooks)
  - [Replication Guide](#replication-guide)

---

## Overview

CLAP (Contrastive Language–Audio Pretraining) aligns audio and text representations in a shared embedding space. Its audio encoder, **HTS-AT**, is a hierarchical Swin-Transformer with four stages of increasing embedding dimension. The final audio embedding is the result of a complex, multi-stage computation — and it is not obvious which attention heads or layers contribute the most to class-discriminative information.

**ResiDual for Audio** is a project in two phases:

1. **Analysis:** Systematically characterise the residual stream of HTS-AT by decomposing attention outputs at the head level, and measuring linear/nonlinear intrinsic dimensionality, class separability, and head specialization across all 184 heads.
2. **Adaptation:** Design spectral reweighting strategies that leverage this characterisation to improve downstream performance, without full fine-tuning.

---

## Project Structure

### Top-level

```
.
├── changes-from-original-clap.txt
├── data/
├── htsat_extraction.drawio.png
├── htsat_pipeline.md
├── intresting-datasets-to-test.txt
├── LICENSE
├── main.py
├── problemi-da-riportare.txt
├── pyproject.toml
├── README.md
├── report/
├── results/
├── src/
├── text.txt
└── uv.lock
```

| Directory / File | Description |
|---|---|
| `report/` | LaTeX source code for the project report. |
| `results/` | Saved outputs from the various notebook runs (metrics, plots, intermediate artefacts). |
| `src/` | Main source tree: ResiDual-CLAP implementation, analysis notebooks, datasets, and model code. |

---

### `src/`

```
src/
├── attention_from_datasets.ipynb
├── CLAPWrapper.py
├── configs/
├── datasets/
├── embeddings/
├── heads_representations/
├── kfolds/
├── models/
├── residual_analysis_pre.ipynb
├── residual_analysis post.ipynb
├── heads_analysis.ipynb
├── kfold_validation.ipynb
├── sanity-check-pca.ipynb
├── test-3.ipynb
└── ...
```

---

#### Configs

`src/configs/` contains the YAML configuration files for both vanilla CLAP and ResiDual-CLAP.

---

#### Datasets

`src/datasets/` contains the Python dataset classes used in the analyses:

| Dataset | Link |
|---|---|
| ESC-50 | *(add link)* |
| IRMAS | *(add link)* |
| TinySOL | *(add link)* |
| VocalSound | *(add link)* |

Ogni codice sorgente di ogni dataset contiene anche una funzione automatica di verifica dei files (verifica di file corrotti) che viene eseguita in automatico all'inizio e i risultati vengono salvati in un file che viene poi ricaricato autometicamente ogni volta che vie e fatto un load del dataset. Se tale file non esiste verrà eseguita di nuovo la validazione.

---

#### Models

`src/models/` contains the CLAP model sources. The files that differ from the original CLAP codebase are:

**Modified original files:**

- `htsat` — the HTS-AT audio encoder; modified to expose internal representations needed by ResiDual.
- `CLAPWrapper.py` — two changes relative to the original:
  - Added `load_residual_clap` function.
  - Modified `load_audio_into_tensor`: uses **deterministic crop of the first 7 seconds** instead of the original random 7-second crop.

**New files added:**

- `residual_clap.py` — core ResiDual-CLAP implementation.
- `residual_clap_utils.py` — utility functions supporting the ResiDual pipeline.
- `train_kfold.py` — k-fold training script.

---

#### K-Folds

`src/kfolds/` stores the k-fold validation results for the four datasets. Note that the definitive results were not run locally but on Kaggle, so this dir does not contain the effective final results; see the notebooks below for the corresponding versions:

| Kaggle Notebook | Layer selection |
|---|---|
| [Classic CLAP vs ResiDualCLAP — K-Fold Validation](https://www.kaggle.com/code/lorenzoarcioni/classic-clap-vs-residualclap-k-fold-validation) (v13) | Layers `{2, 3}` |
| Same notebook (v14) | Layer `{3}` |
| Same notebook (v15) | Layers `{0, 1, 2, 3}` |

---

#### Notebooks

| Notebook | Description |
|---|---|
| `attention_from_datasets.ipynb` | Extracts attention-head activations from HTS-AT running on a dataset (ESC-50) and saves them for downstream analysis. |
| `heads_analysis.ipynb` | Loads the saved attention-head values and analyses their dimensionality using multiple estimators (PCA, TwoNN, MLE, Participation Ratio, Effective Rank, etc.). |
| `residual_analysis_pre.ipynb` / `residual_analysis post.ipynb` | Earlier exploratory code; not central to the current pipeline but kept for reference. |
| `kfold_validation.ipynb` | Runs k-fold validation comparing vanilla CLAP against ResiDual-CLAP across the four datasets. Also in this case, this is **not** the effective notebook used for the final results, the effective one is the Kaggle one above. |
| `sanity-check-pca.ipynb` | Sanity-checks the ResiDual-CLAP implementation, verifying correctness and ruling out unintended behaviours. |
| `test-3.ipynb` | Small local test that exercises ResiDual-CLAP end-to-end and compares its outputs against the original CLAP baseline. |

---

## Replication Guide

> **Coming soon.** A step-by-step guide explaining how to reproduce all experiments will be added here.