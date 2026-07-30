# Multi-Modal AGI LLM Agent Framework

An industry-grade, production-ready Multi-Modal AGI Transformer Language and Vision Model with an agentic reasoning loop — plan, act, reflect, and tool dispatch. Built from scratch with PyTorch for laptop and **Google Colab Nvidia T4 GPU** acceleration.

---

## System Architecture

```
               ┌───────────────────────────┐
               │ Multi-Modal Input Task    │
               │  [Text Prompt + Images]   │
               └─────────────┬─────────────┘
                             ▼
               ┌───────────────────────────┐
               │   Vision Patch Encoder    │ ← ViT-style Patch Embedding + Proj
               └─────────────┬─────────────┘
                             ▼
               ┌───────────────────────────┐
               │  MultimodalLLM Planner    │ ← RoPE + SDPA Flash Attention
               └─────────────┬─────────────┘
                             ▼
               ┌───────────────────────────┐
               │ Multi-Modal Plan Steps    │
               └──────┬──────┬─────────────┘
                      ▼      ▼
               ┌───────────────────────────┐
               │   Multi-Tool Executor     │ ← Vision OCR/Analyzer, Python Sandbox,
               └─────┬─────────────────────┘   Calculator, Web Search, File System
                     ▼
               ┌───────────────────────────┐
               │     Critic Evaluator      │ ← Validates step outputs & retries
               └─────┬─────────────────────┘
                     ▼
               ┌───────────────────────────┐
               │  Episodic Vector Memory   │ ← Similarity search & plan cache
               └───────────────────────────┘
```

### Architecture Specifications

| Component / Parameter | Specification / Technology |
|-----------------------|----------------------------|
| **Base Architecture** | Decoder-Only Multi-Modal Vision-Language Transformer (`MultimodalLLM`) |
| **Vision Encoder**    | ViT Patch Embedding + LayerNorm + Projection Head (`VisionEncoder`) |
| **Positional Encoding**| Rotary Position Embeddings (RoPE) |
| **Attention Engine**  | Scaled Dot-Product Attention (SDPA Flash-Attention) + GQA |
| **Normalizations**    | Root Mean Square Normalization (`RMSNorm`) |
| **Feed-Forward / MLP**| SwiGLU Gated Linear Units |
| **Training Precision**| Automatic Mixed Precision (`torch.cuda.amp.autocast` FP16) |
| **Learning Rate**     | Cosine Annealing Scheduler with Warmup |
| **GPU Acceleration**  | Google Colab Nvidia T4 (16 GB VRAM), CUDA, Apple MPS, CPU |

---

## Quickstart Guide

### 1. Installation & Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Train Tokenizer & Model from Scratch
```bash
# Train the SentencePiece tokenizer
python -m training.tokenizer

# Train the Multi-Modal AGI LLM with AMP mixed precision
python -m training.train
```

### 3. Run the Multi-Modal AGI Agent
```bash
# Run text and multi-modal tasks
python run_agent.py --task "Analyze image data and calculate expression: 15 * (4 + 8)" --image sample.jpg
```

### 4. Google Colab (Nvidia T4 GPU) Training
Open and run [AGI_Multimodal_Colab.ipynb](file:///c:/Users/debnil/projects/agi-mvp-laptop/colab/AGI_Multimodal_Colab.ipynb) on Google Colab or run:
```bash
python -m training.train_colab
```

---

## Project Structure

```
├── agents/             # Agentic loop: Planner, Executor, Critic, Episodic Memory, Multi-modal Tools
├── model/              # PyTorch MultimodalLLM, VisionEncoder, Attention, RMSNorm, SwiGLU, KV-Cache
├── training/           # AMP Training engine, Cosine LR scheduler, Multimodal dataset, Colab scripts
├── colab/              # AGI_Multimodal_Colab.ipynb notebook for Nvidia T4 GPU
├── configs/            # YAML model & training hyper-parameter configurations
├── data/               # Text and image training datasets
├── utils/              # Config loader & helper utilities
├── tests/              # Comprehensive PyTest unit test suite
└── checkpoints/        # Saved model weights
```

---

## Automated Testing

Run the full pytest suite:
```bash
pytest -v
```

---

## License

MIT
