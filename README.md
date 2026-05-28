# ResiDual for Audio: Spectral Reweighting of Residual Streams in CLAP Models

> **Status: Preliminary analysis in progress.**
> This repository documents an ongoing investigation into the internal representations of CLAP's audio encoder (HTS-AT), with the goal of designing spectral reweighting strategies — collectively referred to as **ResiDual** — to improve zero-shot downstream performance.

---

## Table of Contents

- [ResiDual for Audio: Spectral Reweighting of Residual Streams in CLAP Models](#residual-for-audio-spectral-reweighting-of-residual-streams-in-clap-models)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)


---

## Overview

CLAP (Contrastive Language–Audio Pretraining) aligns audio and text representations in a shared embedding space. Its audio encoder, **HTS-AT**, is a hierarchical Swin-Transformer with four stages of increasing embedding dimension. The final audio embedding is the result of a complex, multi-stage computation — and it is not obvious which attention heads or layers contribute the most to class-discriminative information.

**ResiDual for Audio** is a project in two phases:

1. **Analysis:** Systematically characterise the residual stream of HTS-AT by decomposing attention outputs at the head level, and measuring linear/nonlinear intrinsic dimensionality, class separability, and head specialization across all 184 heads.

2. **Adaptation:** Design spectral reweighting strategies that leverage this characterisation to improve downstream performance, without full fine-tuning.

---

