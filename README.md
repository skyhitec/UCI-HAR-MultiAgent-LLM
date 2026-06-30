# On-Device Large Multi-Modal Agent for Human Activity Recognition
### Benchmark Replication & Extension on the UCI HAR Dataset

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badge_svg.svg)](https://skyhitec-uci-har-multiagent-llm-app-vxadvl.streamlit.app/)
[![LangGraph](https://img.shields.io/badge/Agents-LangGraph-1C3C3C.svg)](https://github.com/langchain-ai/langgraph)
[![Model](https://img.shields.io/badge/Base%20Model-LLaMA--3--8B--Instruct-orange.svg)](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Summer Research Internship Project**
> **Student Investigator:** Shudhanshu Yadav — M.Sc. AI & Data Science, DA-IICT, Gandhinagar
> **Faculty Mentor:** Dr. Ankit Vijayvargiya, SMILE Lab, DA-IICT

---

### 🔗 Try it live

The dashboard is deployed and actually usable — not just a screenshot in a README.

**👉 [skyhitec-uci-har-multiagent-llm-app-vxadvl.streamlit.app](https://skyhitec-uci-har-multiagent-llm-app-vxadvl.streamlit.app/)**

You can feed it sensor windows and watch the fine-tuned agent classify the activity, walk through its chain-of-thought reasoning, and hand off to either a sedentary-health advisor or a workout tracker depending on what it predicted.

---

## Table of Contents

- [What this project is about](#what-this-project-is-about)
- [Why this matters — the generalization problem](#why-this-matters--the-generalization-problem)
- [Results at a glance](#results-at-a-glance)
- [How the system works](#how-the-system-works)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Pipeline walkthrough](#pipeline-walkthrough)
- [Screenshots & figures](#screenshots--figures)
- [Tech stack](#tech-stack)
- [Known limitations](#known-limitations)
- [Future work](#future-work)
- [Citation](#citation)
- [License](#license)

---

## What this project is about

This repo replicates and extends the framework from **Siam et al., "On-device Large Multi-modal Agent for Human Activity Recognition"** (arXiv:2512.19742v1). Their original paper benchmarked a QLoRA fine-tuned LLaMA-3-8B agent on three wearable HAR datasets — HHAR, MotionSense, and Shoaib. UCI HAR wasn't part of that evaluation, so that's the gap this project fills.

In short, the pipeline takes raw smartphone IMU sensor data (accelerometer + gyroscope), turns it into structured natural-language descriptions instead of raw numbers, and fine-tunes a large language model to reason over those descriptions and classify human activity — while staying robust to subjects it has never seen before.

If you've worked with HAR before, you already know the punchline: classical models (SVM, RF, DNN, LSTM) are great at classifying activities for people they were trained on, and fall apart the moment you test them on a new person. That's the problem this project is actually trying to solve.

---

## Why this matters — the generalization problem

Train an SVM, Random Forest, DNN, or LSTM on a HAR dataset and test it on the same subjects, and you'll comfortably clear 90%+ F1. Test that same model on subjects it has never seen, and the score can fall by 40–50 percentage points. This happens because these models end up memorizing subject-specific biometric signatures — how *this particular person* walks, how the sensor sat on *their* hip — rather than learning what walking actually looks like in general.

The hypothesis behind this project: a fine-tuned LLM, because it starts with broad pre-trained knowledge about how humans move, can reason over *semantic descriptions* of sensor patterns instead of memorizing raw numeric thresholds. That should make it generalize better to people it's never seen.

The results below back that up.

---

## Results at a glance

### Macro F1-Score — Seen vs. Unseen Subjects

| Model | HHAR | MotionSense | Shoaib | **UCI HAR (this repo)** |
|:---|:---:|:---:|:---:|:---:|
| SVM | 0.910 / 0.470 | 0.820 / 0.420 | 0.920 / 0.560 | 0.954 / 0.939 |
| Random Forest | 0.970 / 0.670 | 0.840 / 0.580 | 0.980 / 0.670 | 0.925 / 0.915 |
| DNN | 0.980 / 0.610 | 0.810 / 0.380 | 0.970 / 0.640 | 0.980 / 0.580 |
| LLaMA-3-8B (Zero-Shot) | — / 0.650 | — / 0.610 | — / 0.670 | — / 0.630 |
| **LLaMA-3-8B (Fine-Tuned, Ours)** | 0.830 / **0.750** | 0.770 / **0.650** | 0.790 / **0.710** | 0.785 / **0.801** |

*(format: Seen F1 / Unseen F1)*

A few things worth pointing out here:

- The DNN gets the best seen-subject score on UCI HAR (0.980), but collapses to 0.580 on unseen subjects — a 40-point drop.
- The fine-tuned LLaMA-3 agent is the **only model in the table whose unseen score beats its seen score** (0.785 → 0.801). That's not something you'd normally expect, and it's a strong signal that the model is actually reasoning about activity patterns rather than memorizing subject quirks.
- On every single dataset, fine-tuned LLaMA-3 wins on the unseen split. The pattern isn't a fluke of one dataset — it holds across HHAR, MotionSense, Shoaib, and now UCI HAR.

### Headline numbers

| | |
|---|---|
| **Unseen F1 (UCI HAR)** | **0.801** |
| Improvement over zero-shot LLaMA-3 | **+17.1%** |
| RAG instruction pairs generated | 20,264 |
| Multi-task objectives covered | 4 (Classification, CoT, Posture QA, Health Recs) |
| Base model | Meta-LLaMA-3-8B-Instruct |
| Fine-tuning method | QLoRA, rank 128, 4-bit NF4 |

---

## How the system works

The pipeline has four stages, end to end:

```
Raw IMU Sensor Data (UCI HAR)
        │
        ▼
┌────────────────────────┐
│ 1. Features-to-Text     │  561-dim feature vector → structured
│    Serialization        │  natural language description
└────────────┬────────────┘
             ▼
┌────────────────────────┐
│ 2. RAG-Guided           │  FAISS-indexed biomechanical knowledge
│    Retrieval             │  base → grounds predictions in physical rules
└────────────┬────────────┘
             ▼
┌────────────────────────┐
│ 3. Multi-Task Prompt    │  Classification / CoT / QA / Health Recs
│    Assembly              │  assembled with LLaMA-3 chat templates
└────────────┬────────────┘
             ▼
┌────────────────────────┐
│ 4. QLoRA Fine-Tuning     │  LoRA adapters injected into LLaMA-3-8B,
│                          │  trained on 20,264 instruction pairs
└────────────┬────────────┘
             ▼
   Fine-Tuned HAR Agent
             │
             ▼
┌────────────────────────────────────────────┐
│         LangGraph Multi-Agent Layer          │
│                                               │
│  Input → Classifier+RAG → Router             │
│                          ├─→ Sedentary Advisor│
│                          └─→ Workout Tracker  │
│                              ↓                │
│                       Memory Formatter        │
└─────────────────────┬────────────────────────┘
                       ▼
            Streamlit Dashboard (live)
```

### Why text, not raw numbers?

A 128-sample window across 9 sensor channels is 1,152 raw floating-point values — way too much to stuff into a prompt, and raw numbers give the model nothing to reason about anyway. Instead, the pipeline pulls a focused subset of interpretable features (per-channel mean, std, spectral entropy, dominant frequency) out of the official 561-dimensional UCI HAR feature vector and serializes them into something like:

> *"Accelerometer body mean: [0.24, -0.025, 0.98], std: [0.41, 0.21, 0.32]; Gyroscope mean: [0.01, -0.03, 0.02]; spectral entropy: 3.42; dominant frequency: 1.8 Hz"*

That description is then grounded against a small hand-built knowledge base of biomechanical patterns (via FAISS retrieval using `all-MiniLM-L6-v2` embeddings) before being handed to the model — which is what keeps the model from hallucinating activity labels on edge cases.

### The 9-files-vs-6-channels thing

If you've downloaded raw UCI HAR before, you've probably noticed the `Inertial Signals/` folder has **9** files per split, even though there are only **2 physical sensors** (accelerometer + gyroscope = 6 motion channels). That's because the dataset authors ran a Butterworth filter over the raw accelerometer signal to split it into body acceleration and gravity components:

```
a_body(t) = a_total(t) - a_gravity(t)
```

So you get `total_acc_{x,y,z}`, `body_acc_{x,y,z}`, and `body_gyro_{x,y,z}` — 9 files, 6 real physical channels. This pipeline uses all 9 for the LSTM baseline (it benefits from the gravity orientation info) but sticks to the 6 body-motion channels for LLM serialization to keep things physically interpretable.

### Why 128-sample windows?

At 50 Hz, a 128-sample window covers 2.56 seconds — long enough to capture at least two full gait cycles (a normal walking stride is roughly 1.0–1.4s), giving the FFT enough context to resolve stride frequency without making the window so long that real-time inference becomes impractical.

### The LangGraph multi-agent layer

Classification alone isn't that interesting for a real wearable assistant — what matters is what happens *after* you know the activity. So this project wraps the fine-tuned model in a 6-node stateful LangGraph pipeline:

1. **Input Validation** — sanity-checks the incoming feature dict
2. **LLaMA-3 Classifier + RAG** — predicts activity, confidence, and CoT reasoning
3. **Conditional Router** — static postures (Sitting/Standing/Laying) go one way, dynamic activities (Walking variants) go another
4. **Sedentary Advisor** — health nudges for prolonged inactivity
5. **Workout Tracker** — estimates step rate, intensity, and calorie burn for active movement
6. **Memory Formatter** — packages everything into a structured JSON output

The whole thing runs on a shared `HARAgentState` so the reasoning trace, retrieved context, and routing decision all stay auditable end to end.

---

## Repository structure

```
UCI-HAR-MultiAgent-LLM/
│
├── final_pipeline.py                  # 1800+ lines — full pipeline: signal
│                                       # processing → baselines → RAG dataset
│                                       # generation → QLoRA SFT config
│
├── langgraph_har_pipeline.py          # 6-node LangGraph multi-agent system
│
├── demo_inference.py                  # Multi-task inference demo —
│                                       # classification, CoT, QA, health recs
│
├── app.py                             # Streamlit dashboard (live deployed)
│
├── generate_pdf_report.py             # ReportLab IEEE-format PDF generator
├── generate_architecture_diagram.py   # Architecture figure generation
├── generate_extra_plots.py            # Additional EDA/results plots
├── generate_consolidated_table.py     # Cross-dataset comparison table
├── generate_four_datasets_plot.py     # Multi-dataset comparison chart
│
├── instruction_dataset.json           # 20,264 RAG-guided instruction pairs
├── feature_names.txt                  # 561 official UCI HAR feature names
│
├── X_train_feat.npy / X_test_feat.npy # Saved feature arrays
├── y_train.npy / y_test.npy           # Saved labels
├── subjects_train.npy / subjects_test.npy
│
├── UCI HAR Dataset/                   # Raw dataset (Inertial Signals/, etc.)
├── fig*.png                           # 18 publication-quality figures
│
├── requirements.txt
└── LICENSE
```

---

## Getting started

### 1. Clone and set up the environment

```bash
git clone https://github.com/skyhitec/UCI-HAR-MultiAgent-LLM.git
cd UCI-HAR-MultiAgent-LLM

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python final_pipeline.py
```

This loads the dataset, runs feature extraction, trains the baseline classifiers, generates the RAG instruction dataset, and sets up the QLoRA fine-tuning configuration.

### 3. Try the multi-task inference demo

```bash
python demo_inference.py
```

### 4. Launch the dashboard locally

```bash
streamlit run app.py
```

> **Note:** QLoRA fine-tuning of an 8B model needs a real GPU. The training in this project was run on Lightning AI cloud infrastructure — 4-bit NF4 quantization brings the memory footprint down from ~16 GB (BF16) to under 6 GB, which is what makes this feasible outside a multi-GPU server.

---

## Pipeline walkthrough

| Stage | What happens | Output |
|---|---|---|
| **1. Data ingestion** | Load 9 raw Inertial Signal files + 561 official features | `X_train_feat.npy`, etc. |
| **2. Baselines** | Train SVM, RF, DNN, LSTM on seen split, re-evaluate on unseen split | F1 comparison tables |
| **3. RAG dataset gen** | Serialize features → text, retrieve biomechanical context, assemble 4 task types | `instruction_dataset.json` |
| **4. QLoRA fine-tuning** | Fine-tune LLaMA-3-8B-Instruct on 20,264 pairs, 3 epochs | `llama3_har_finetuned/final` adapter weights |
| **5. Multi-agent system** | Wrap fine-tuned model in LangGraph 6-node pipeline | Structured JSON predictions |
| **6. Dashboard** | Deploy interactive Streamlit app | Live demo |

---

## Screenshots & figures

**Per-activity sensor signals** — accelerometer + gyroscope traces for all 6 UCI HAR activities:

<img width="900" alt="all activity signals" src="https://github.com/user-attachments/assets/3640a940-d284-4ad2-8693-d67cef5a3401" />

**Seen vs. unseen generalization** — the chart that this whole project is really about:

<img width="700" alt="seen vs unseen" src="https://github.com/user-attachments/assets/e44f448b-f4bc-4d24-98c8-38f98f81d01d" />

**Fine-tuning pipeline architecture:**

<img width="500" alt="finetuning pipeline" src="https://github.com/user-attachments/assets/c1006d2a-c1a2-4fe1-ac04-34353047ebec" />

**Complete project architecture** — Data Layer → Model Layer → Application Layer:

<img width="500" alt="complete architecture" src="https://github.com/user-attachments/assets/0ca07903-123f-4cf7-885a-20dd7b78c64a" />

---

## Tech stack

| Layer | Tools |
|---|---|
| **Modeling** | Meta-LLaMA-3-8B-Instruct, QLoRA / PEFT, bitsandbytes (4-bit NF4) |
| **RAG** | sentence-transformers (`all-MiniLM-L6-v2`), FAISS (`IndexFlatL2`) |
| **Orchestration** | LangChain, LangGraph |
| **Baselines** | scikit-learn (SVM, Random Forest), TensorFlow/Keras (DNN, LSTM) |
| **Dashboard** | Streamlit |
| **Reporting** | ReportLab (automated IEEE-format PDF generation) |
| **Training infra** | Lightning AI (cloud GPU) |

---

## Known limitations

- The 4,096-token context limit on LLaMA-3-8B means only a curated subset of the 561 features actually makes it into each prompt — there's room to lose some signal here.
- Fine-tuned adapter weights aren't checked into this repo (they're large) — you'll need to either retrain via `final_pipeline.py` or source them separately.
- The LangGraph `StateGraph` compilation has version-sensitivity issues across LangChain/LangGraph releases; a procedural execution fallback is included that preserves the same node logic if the native graph compile fails.
- Subjects 25–30 in UCI HAR happen to show fairly consistent gait rhythms, which may be part of why unseen F1 slightly exceeds seen F1 here specifically — worth keeping in mind when comparing across datasets.

---

## Future work

- Swap the manual features-to-text serialization for a learned sensor encoder (something SignalCLIP-style) to remove the hand-picked feature selection step.
- Quantize the fine-tuned adapters with `llama.cpp` or MLC-LLM and actually test inference on ARM wearable hardware — the on-device part of "on-device agent" hasn't been validated yet.
- Cross-dataset transfer: train on UCI HAR, test on WISDM, to get a harder test of out-of-distribution generalization than within-dataset unseen subjects.
- Real-time streaming inference with a sliding window buffer, so the dashboard can take live sensor input instead of static test samples.

---

## Citation

This project replicates and extends:

```bibtex
@article{siam2025ondevice,
  title={On-device Large Multi-modal Agent for Human Activity Recognition},
  author={Siam, M. S. I. and Showmik, I. A. and Song, G. and Zhu, T.},
  journal={arXiv preprint arXiv:2512.19742v1},
  year={2025}
}
```

UCI HAR Dataset:

```bibtex
@inproceedings{anguita2013public,
  title={A public domain dataset for human activity recognition using smartphones},
  author={Anguita, D. and Ghio, A. and Oneto, L. and Parra, X. and Reyes-Ortiz, J. L.},
  booktitle={Proc. ESANN},
  pages={437--442},
  year={2013}
}
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<p align="center">
Built during a Summer Research Internship at SMILE Lab, DA-IICT.<br>
Questions or issues? Open one on this repo.
</p>
