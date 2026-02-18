
---

# 📦 requirements.txt

```txt
torch
torchvision
timm
numpy
matplotlib
scikit-learn
umap-learn
pyyaml



# 🦴 Orthopedic Foundation Model (Self-Supervised Learning)

This repository demonstrates a self-supervised foundation model pipeline for orthopedic X-ray analysis.

The goal is to learn hospital-robust representations from large-scale unlabeled X-ray datasets and transfer them to downstream clinical tasks.

---

## 🔬 Motivation

Orthopedic imaging datasets often:

- Contain millions of X-rays
- Span multiple hospitals
- Have limited or delayed labels
- Include multiple images per patient

Self-supervised learning (SSL) enables scalable representation learning without requiring manual annotation.

---

## 🧠 Pipeline Overview

1. Self-supervised pretraining (SimCLR-style)
2. Embedding extraction
3. Label-free kNN retrieval
4. Patient-level aggregation (mean + attention pooling)
5. Linear probe evaluation

---

## 📊 Example Results

| Component | Description |
|-----------|------------|
| UMAP | Visualizes embedding structure |
| kNN retrieval | Similar-case retrieval |
| Attention pooling | Patient-level modeling |
| Linear probe | Downstream separability |

---

## 🏗 Repository Structure

See `src/` for modular code.

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python scripts/run_pretraining.py
python scripts/run_evaluation.py
