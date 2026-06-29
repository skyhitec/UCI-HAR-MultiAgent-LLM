# On-Device Large Multi-Modal Agent for Human Activity Recognition (UCI HAR Extension)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badge_svg.svg)](https://skyhitec-uci-har-multiagent-llm-app-vxadvl.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Summer Research Internship Project**  
> **Student Investigator:** Shudhanshu Yadav  
> **Institution:** Central University of Andhra Pradesh & DA-IICT  

---

## 🌐 Live Interactive Dashboard Demo
🎮 **Access the Live Web Application here:**  
👉 **[https://skyhitec-uci-har-multiagent-llm-app-vxadvl.streamlit.app/](https://skyhitec-uci-har-multiagent-llm-app-vxadvl.streamlit.app/)**

The live Streamlit dashboard provides interactive 6-channel IMU telemetry visualization, real-time activity predictions, and step-by-step Chain-of-Thought (CoT) multi-agent reasoning logs.

---

## 📌 Abstract & Overview

This repository contains the complete benchmark replication and extension codebase for the research paper:  
**"On-device Large Multi-modal Agent for Human Activity Recognition"** (*Siam et al., arXiv:2512.19742v1*).

While the original paper focused on HHAR, MotionSense, and Shoaib datasets, this project extends the framework to the benchmark **UCI HAR Dataset** (30 subjects, 6 physical activities, 561 statistical features). We implement a Retrieval-Augmented Generation (RAG) guided instruction dataset generation pipeline (20,264 prompt-response pairs), parameter-efficient fine-tuning (QLoRA) for Meta-LLaMA-3-8B-Instruct, and a stateful multi-agent LangGraph orchestration workflow.

---

## 🌟 Key Features & Contributions

- 🐍 **Complete Replication Pipeline (`final_pipeline.py`):** 1800+ lines of modular Python code implementing end-to-end signal filtering, 561-feature extraction, baseline classifiers (SVM, RF, DNN, LSTM), and QLoRA SFT configuration.
- 🤖 **Interactive Multi-Task Demo (`demo_inference.py`):** 430+ line inference script showcasing classification, step-by-step Chain-of-Thought (CoT) reasoning, posture QA, and personalized health recommendations.
- 📊 **Glassmorphism Web Dashboard (`app.py`):** 600+ line interactive Streamlit application deployed live at [Streamlit Cloud](https://skyhitec-uci-har-multiagent-llm-app-vxadvl.streamlit.app/).
- 🔄 **LangChain & LangGraph Orchestration (`langgraph_har_pipeline.py`):** 6-node stateful agent architecture with conditional routing and dynamic memory.
- 📑 **IEEE PDF Generator (`generate_pdf_report.py`):** Automated compilation script generating the 10-page academic technical report (`HAR_Project_Report.pdf`).

---

## 📊 Benchmark Generalization Matrix (Macro F1-Score)

| Model Architecture | HHAR (Seen / Unseen) | MotionSense (Seen / Unseen) | Shoaib (Seen / Unseen) | UCI HAR (Ours) (Seen / Unseen) |
|:---|:---:|:---:|:---:|:---:|
| **Support Vector Machine (SVM)** | 0.910 / 0.470 | 0.820 / 0.420 | 0.920 / 0.560 | 0.954 / 0.939 |
| **Random Forest (RF)** | 0.970 / 0.670 | 0.840 / 0.580 | 0.980 / 0.670 | 0.925 / 0.915 |
| **Deep Neural Network (DNN)** | 0.980 / 0.610 | 0.810 / 0.380 | 0.970 / 0.640 | 0.980 / 0.580 |
| **LLaMA-3-8B (Base Zero-Shot)** | - / 0.650 | - / 0.610 | - / 0.670 | - / 0.630 |
| **LLaMA-3-8B (Fine-Tuned Ours)** | 0.830 / **0.750** | 0.770 / **0.650** | 0.790 / **0.710** | 0.785 / **0.801** |

---

## 🚀 Getting Started

### 1. Installation & Environment Setup
```bash
# Clone the repository
git clone https://github.com/skyhitec/UCI-HAR-MultiAgent-LLM.git
cd UCI-HAR-MultiAgent-LLM

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Full Pipeline
```bash
python final_pipeline.py
```

### 3. Launching the Interactive Web Dashboard Locally
```bash
streamlit run app.py
```

---

## Result and Working Pipelines
<img width="2684" height="1476" alt="fig7_all_activity_signals" src="https://github.com/user-attachments/assets/3640a940-d284-4ad2-8693-d67cef5a3401" />

<img width="2234" height="931" alt="fig4_seen_vs_unseen" src="https://github.com/user-attachments/assets/e44f448b-f4bc-4d24-98c8-38f98f81d01d" />
<img width="1024" height="1024" alt="fig17_finetuning_pipeline_architecture" src="https://github.com/user-attachments/assets/c1006d2a-c1a2-4fe1-ac04-34353047ebec" />
<img width="1024" height="1024" alt="fig18_complete_project_architecture" src="https://github.com/user-attachments/assets/0ca07903-123f-4cf7-885a-20dd7b78c64a" />

---


---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
