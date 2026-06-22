# 🔺 Project Trinity-AI

> An end-to-end Artificial Intelligence portfolio ecosystem — three complementary pillars covering the core verticals demanded by the modern AI market: **Generative NLP**, **Computer Vision**, and **Classical Machine Learning**.

Built locally in Python on a single **NVIDIA RTX 3060 (12 GB)**, with deliberate emphasis on the *engineering and mathematics* behind each system — not just calling high-level APIs.

---

## 🎯 Philosophy

This is not a collection of tutorials. Each pillar is built to demonstrate **deep, first-principles understanding** of how modern AI systems actually work — from raw tensors and gradients up to deployed inference. Special attention is paid to the real-world constraint of training on **consumer-grade hardware (12 GB VRAM)**.

---

## 🏛️ The Three Pillars

### 1️⃣ SLM — Small Language Model from Scratch
A compact, GPT-2-style language model implemented in **pure PyTorch**, trained on the scientific domain (generating paper abstracts from academic titles).

- Custom **Byte-Pair Encoding (BPE)** tokenizer, built from scratch
- Transformer architecture: multi-head self-attention, positional embeddings, residual streams
- Memory engineering for 12 GB VRAM: **gradient accumulation** + **mixed-precision (AMP)** training
- 📂 [`01-slm-from-scratch/`](./01-slm-from-scratch/)

### 2️⃣ Applied Computer Vision
Real-time object detection (YOLO family) with a focus on the full data-to-inference pipeline.

- Tensor manipulation and data-augmentation pipelines (**Albumentations**)
- Live webcam inference using **OpenCV**
- 📂 [`02-computer-vision/`](./02-computer-vision/)

### 3️⃣ Churn Prediction — Classical ML
A business-oriented tabular prediction system, judged by metrics that matter to decision-makers.

- Exploratory Data Analysis (EDA) and advanced feature engineering
- Class-imbalance handling (**SMOTE**)
- Tree-based models (**Random Forest / XGBoost**) evaluated by **Recall** and **ROC-AUC**
- 📂 [`03-churn-prediction/`](./03-churn-prediction/)

---

## 🧰 Tech Stack

| Domain | Tools |
|--------|-------|
| Core | Python, NumPy, Pandas |
| Deep Learning | PyTorch, CUDA, AMP |
| NLP | Custom BPE tokenizer |
| Computer Vision | OpenCV, Albumentations, YOLO |
| Classical ML | scikit-learn, XGBoost, imbalanced-learn |
| Hardware | NVIDIA RTX 3060 (12 GB) |

---

## 📁 Repository Structure

```
Trinity-AI/
├── 01-slm-from-scratch/      # Pillar 1 — Generative NLP
├── 02-computer-vision/       # Pillar 2 — Object Detection
├── 03-churn-prediction/      # Pillar 3 — Classical ML
├── .gitignore
└── README.md
```

Each pillar is **self-contained**: its own virtual environment, dependencies, and documentation.

---

## 🚧 Status

| Pillar | Status |
|--------|--------|
| 1 — SLM from Scratch | 🟡 In progress |
| 2 — Computer Vision | ⚪ Planned |
| 3 — Churn Prediction | ⚪ Planned |

---

*Built as a deliberate learning journey through the full AI engineering stack.*
