<div align="center">

# 🧠⚡ Multi-Modal AGI LLM & Agentic Framework

### *From-Scratch PyTorch Vision-Language Transformer & Autonomous Agent Reasoning Loop*

[![CI Build](https://github.com/Debddj/agi-mvp-laptop/actions/workflows/ci.yml/badge.svg)](https://github.com/Debddj/agi-mvp-laptop/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Debddj/agi-mvp-laptop/blob/main/colab/AGI_Multimodal_Colab.ipynb)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

**An industry-grade, production-ready Multi-Modal AGI Transformer with an autonomous Agentic Reasoning Loop (Plan ➔ Act ➔ Reflect ➔ Tool Dispatch). Built from scratch in PyTorch for local deployment and Google Colab Nvidia T4 GPU acceleration.**

[Key Features](#-key-features) • [Architecture](#-system-architecture) • [Quickstart](#-quickstart-guide) • [Google Colab](#-google-colab-t4-gpu) • [Project Structure](#-project-structure) • [License](#-license)

---

</div>

## 🌟 Key Features

| Feature | Description |
| :--- | :--- |
| 👁️ **Multi-Modal Vision Encoder** | ViT-style patch projection layer with LayerNorm and projection heads for joint vision-text processing. |
| ⚡ **SDPA Flash Attention & RoPE** | Rotary Position Embeddings (RoPE) and PyTorch Scaled Dot-Product Attention for high performance. |
| 🧠 **Autonomous Agentic Loop** | Closed-loop **Planner ➔ Executor ➔ Critic ➔ Reflect** reasoning system with dynamic retries. |
| 💾 **KV-Cache Acceleration** | Efficient auto-regressive decoding via key-value caching across transformer layers. |
| 🧰 **Multi-Tool Suite** | Built-in Python Sandbox, Vision OCR/Analyzer, Math Evaluator, Web Search, and File I/O tools. |
| 📚 **Episodic Vector Memory** | Semantic plan indexing & caching to reuse successful solution trajectories. |
| 🚀 **Google Colab GPU Ready** | One-click Google Colab notebook for fine-tuning on Nvidia T4 VRAM (16 GB). |

---

## 🏗️ System Architecture

```
                       ┌───────────────────────────┐
                       │  Multi-Modal Input Task   │
                       │   [Text Prompt + Image]   │
                       └─────────────┬─────────────┘
                                     ▼
                       ┌───────────────────────────┐
                       │   Vision Patch Encoder    │ ← ViT Patch Embedding + Proj
                       └─────────────┬─────────────┘
                                     ▼
                       ┌───────────────────────────┐
                       │  MultimodalLLM Planner    │ ← RoPE + SDPA Flash Attention
                       └─────────────┬─────────────┘
                                     ▼
                       ┌───────────────────────────┐
                       │  Multi-Modal Plan Steps   │
                       └──────┬──────┬─────────────┘
                              ▼      ▼
                       ┌───────────────────────────┐
                       │   Multi-Tool Executor     │ ← Vision OCR, Python Sandbox,
                       └─────┬─────────────────────┘   Calculator, Web Search
                             ▼
                       ┌───────────────────────────┐
                       │     Critic Evaluator      │ ← Validates step outputs & retries
                       └─────┬─────────────────────┘
                             ▼
                       ┌───────────────────────────┐
                       │  Episodic Vector Memory   │ ← Plan caching & vector similarity
                       └───────────────────────────┘
```

### 🔬 Technical Specifications

- **Transformer Architecture**: Decoder-Only Multi-Modal Vision-Language Model (`MultimodalLLM`)
- **Vision Encoding**: ViT Patch Embedding + LayerNorm + Linear Projection Head (`VisionEncoder`)
- **Positional Encoding**: Rotary Position Embeddings (`RoPE`)
- **Attention Engine**: Scaled Dot-Product Attention (`SDPA`) + Grouped Query Attention (`GQA`)
- **Norm & MLP**: Root Mean Square Normalization (`RMSNorm`) & SwiGLU Gated Linear Units
- **Precision & Training**: Automatic Mixed Precision (`torch.cuda.amp.autocast` FP16) + Cosine LR Scheduler

---

## ⚡ Quickstart Guide

### 1️⃣ Installation & Environment Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/Debddj/agi-mvp-laptop.git
cd agi-mvp-laptop
pip install -r requirements.txt
```

### 2️⃣ Train Model & Tokenizer from Scratch

```bash
# 1. Train the SentencePiece tokenizer on custom corpus
python -m training.tokenizer

# 2. Train the Multi-Modal AGI LLM model with PyTorch AMP FP16
python -m training.train
```

### 3️⃣ Run the Agentic Reasoning Loop

Execute complex multi-modal tasks using the interactive launcher:

```bash
python run_agent.py --task "Analyze image data and calculate expression: 15 * (4 + 8)" --image sample.jpg
```

#### 📋 Example Execution Output

```
==================================================
        MULTI-MODAL AGI AGENT REASONING LOOP      
==================================================
Task: Analyze image data and calculate expression: 15 * (4 + 8)
Image: sample.jpg

[Step 1] Planning task breakdown...
  ├─ Action: Run Vision Analyzer on sample.jpg
  ├─ Action: Run Python Sandbox for calculation 15 * (4 + 8)
[Step 2] Executing Tools...
  ├─ Tool [vision_analyzer]: Extracted image features successfully.
  ├─ Tool [python_sandbox]: Result = 180
[Step 3] Critic Evaluation...
  └─ Status: VERIFIED PASSED (Confidence: 1.0)

[✓] Final Answer: 180
==================================================
```

---

## ☁️ Google Colab (Nvidia T4 GPU)

Train and benchmark the model directly on Google Colab with free Nvidia T4 VRAM acceleration:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Debddj/agi-mvp-laptop/blob/main/colab/AGI_Multimodal_Colab.ipynb)

Or run the Colab automation script:

```bash
python -m training.train_colab
```

---

## 📁 Project Structure

```
agi-mvp-laptop/
├── 🤖 agents/            # Autonomous Agent Loop (Planner, Executor, Critic, Memory, Tools)
├── 🧠 model/             # PyTorch MultimodalLLM, VisionEncoder, RoPE Attention, KV-Cache, SwiGLU
├── 🏋️ training/          # AMP FP16 Training Engine, SentencePiece Tokenizer, Colab Scripts
├── 📓 colab/             # AGI_Multimodal_Colab.ipynb notebook for Nvidia T4 GPU
├── ⚙️ configs/           # Model & Training hyper-parameter configurations (YAML)
├── 📊 data/              # Text and image training datasets
├── 🧪 tests/              # Comprehensive PyTest unit test suite (27 passing tests)
├── 🛠️ utils/              # Helper utilities & configuration loaders
└── 📄 pyproject.toml     # Build system & ruff/pytest configurations
```

---

## 🧪 Automated Testing

Run the full automated test suite:

```bash
pytest -v
```

---

## 🛡️ License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
