# ResiDual for Audio: Spectral Reweighting of Residual Streams in CLAP Models

> **Status: Preliminary analysis in progress.**
> This repository documents an ongoing investigation into the internal representations of CLAP's audio encoder (HTS-AT), with the goal of designing spectral reweighting strategies — collectively referred to as **ResiDual** — to improve zero-shot downstream performance.

---

## Table of Contents

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

1. **Analysis:** Systematically characterise the residual stream of HTS-AT by decomposing attention outputs at the head level, and measuring linear/nonlinear intrinsic dimensionality, class separability, and head specialisation across all 184 heads.
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
├── interesting-datasets-to-test.txt
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
├── residual_analysis_post.ipynb
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
| ESC-50 | [link](https://github.com/karoldvl/ESC-50/archive/master.zip) |
| IRMAS | [link](https://zenodo.org/records/1290750/files/IRMAS-TrainingData.zip?download=1) |
| TinySOL | [link](https://zenodo.org/record/3685367/files/TinySOL.tar.gz) |
| VocalSound | [link](https://huggingface.co/datasets/lorenzo-arcioni/vocalsound-test/resolve/main/VocalSound.zip) |

Each dataset module includes an automatic file integrity check that runs on first load. Results are cached to disk and reloaded automatically on subsequent runs; if the cache file is missing, validation is re-executed.

> **Note on VocalSound:** The effective download link for the original dataset is `https://www.dropbox.com/s/ybgaprezl8ubcce/vs_release_44k.zip?dl=1`. Due to intermittent Dropbox download failures, the dataset was re-uploaded to HuggingFace, and the HuggingFace link above should be used instead.

---

#### Models

`src/models/` contains the CLAP model sources. The files that differ from the original CLAP codebase are:

**Modified original files:**

- `htsat` — the HTS-AT audio encoder; modified to expose internal representations needed by ResiDual.
- `CLAPWrapper.py` — two changes relative to the original:
  - Added `load_residual_clap` function.
  - Modified `load_audio_into_tensor`: uses a **deterministic crop of the first 7 seconds** instead of the original random 7-second crop, to ensure reproducible behaviour across runs.

**New files added:**

- `residual_clap.py` — core ResiDual-CLAP implementation.
- `residual_clap_utils.py` — utility functions supporting the ResiDual pipeline.
- `train_kfold.py` — k-fold training script.

---

#### K-Folds

`src/kfolds/` stores the k-fold validation results for the four datasets. Note that the definitive results were not run locally but on Kaggle, so this directory does not contain the final results. Refer to the Kaggle notebooks below for the corresponding versions:

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
| `residual_analysis_pre.ipynb` / `residual_analysis_post.ipynb` | Earlier exploratory code; not central to the current pipeline but kept for reference. |
| `kfold_validation.ipynb` | Runs k-fold validation comparing vanilla CLAP against ResiDual-CLAP across the four datasets. This is **not** the notebook used for the final results reported in the paper; see the Kaggle notebook linked above. |
| `sanity-check-pca.ipynb` | Sanity-checks the ResiDual-CLAP implementation, verifying correctness and ruling out unintended behaviours. |
| `test-3.ipynb` | Small local test that exercises ResiDual-CLAP end-to-end and compares its outputs against the original CLAP baseline. |

---

## Replication Guide

> **Requirements:** At least 8 GB of available RAM is required; 16 GB or more is recommended for comfortable headroom.

### Cloning and installing

```bash
git clone https://github.com/lorenzo-arcioni/ResiDual-CLAP.git
cd ResiDual-CLAP
uv sync
```

Select the correct environment before running any of the notebooks below.

*Note:* ALL local experiment (and so also the entire uv environment) use Pytorch CPU only version (I don't have a GPU). Thus all the local experiments are run in the `cpu` environment.

---

### Replicating the analysis

> Note: each analysis notebook also contains additional plots and figures beyond those included in the paper, as part of a more in-depth exploration.

1. Open `attention_from_datasets.ipynb`.
2. Run only the **first 4 Python cells**, up to and including the cell titled *Extractor 1: Per-head representations*.
3. Open `heads_analysis.ipynb` and run all cells to reproduce all graphs and plots used in the paper.

> Before running `heads_analysis.ipynb`, ensure sufficient RAM is available. If needed, clear the memory used by the previous notebook first.

---

### Running the ResiDual functional test

This notebook tests both CLAP versions on zero-shot accuracy on ESC-50 (other datasets can be substituted after downloading them).

1. Open `test-3.ipynb` and run all cells.

> Before running, ensure sufficient RAM is available and clear previous notebook memory if needed.

**Expected results:**
- Original CLAP accuracy: **93.3%**
- ResiDual-CLAP accuracy: **95.3%**

This is a functional smoke test to confirm the code runs correctly before proceeding to full k-fold validation.

---

### Replicating k-fold validation (CPU / no GPU)

This notebook is a local functional test to verify that the k-fold pipeline runs correctly before the full experiment on Kaggle.

1. Open `kfold_validation.ipynb` and run all cells.

> Before running, ensure sufficient RAM is available and clear previous notebook memory if needed.

---

### Replicating k-fold validation (with GPU, full experiment)

1. Open the Kaggle notebook at: https://www.kaggle.com/code/lorenzoarcioni/classic-clap-vs-residualclap-k-fold-validation (version 13).
2. Copy and edit the notebook to run the full experiment yourself.
3. **Enable the T4 GPU accelerator** before running (the P100 has known compatibility issues with the Kaggle PyTorch environment).

All required libraries are installed via the `pip` command at the top of the notebook.