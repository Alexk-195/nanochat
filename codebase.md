# nanochat Project Guide

## Project Overview

**nanochat** is the simplest experimental harness for training and deploying LLMs end-to-end on a single GPU node. It covers all major LLM stages: tokenization, pretraining, finetuning, evaluation, inference, and a web UI for chatting.

**Key Achievement**: Train a GPT-2 capability LLM in ~3 hours on an 8XH100 GPU node for ~$48 (vs. $43,000 in 2019).

**Core Design Philosophy**:
- Minimal, hackable, readable code (~3000 lines)
- Single complexity dial: `--depth` (transformer layers) automatically optimizes all hyperparameters
- Compute-optimal models at any scale (depth 26 ≈ GPT-2, depth 12 ≈ GPT-1)
- No giant config objects, factory functions, or excessive conditionals

---

## Quick Start

### Train a GPT-2 Model
```bash
bash runs/speedrun.sh  # ~3 hours on 8XH100 GPU
```

### Chat with Your Model
```bash
python -m scripts.chat_web  # Open web UI at http://localhost:8000
```

### Quick Experimentation (5-10 min runs)
```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train \
    --depth=12 --run="d12" --model-tag="d12"
```

---

## Directory Structure & File Purposes

### Root Level
- **`README.md`** - Main documentation with leaderboard, guides, and overview
- **`pyproject.toml`** - Python project configuration, dependencies (PyTorch, wandb, etc.)
- **`uv.lock`** - Lock file for reproducible Python environments (uses `uv` package manager)
- **`LICENSE`** - MIT License
- **`.python-version`** - Python version specification (for pyenv/uv)

### `nanochat/` - Core Library

**Architecture & Model**
- **`gpt.py`** - Main GPT transformer model definition (nn.Module)
  - Implements: embeddings, attention, MLPs, transformer blocks
  - Uses `--depth` to auto-calculate: width, num_heads, layer counts, etc.

**Data & Tokenization**
- **`dataset.py`** - Download and read pretraining data (Fineweb, ClimbMix, etc.)
  - Handles dataset versioning, sharding, caching

- **`dataloader.py`** - Distributed tokenizing data loader
  - Real-time BPE tokenization during training
  - Supports multi-GPU distributed loading via DDP

- **`tokenizer.py`** - BPE tokenizer wrapper (GPT-4 style)
  - Encoding/decoding of text ↔ tokens

**Training & Optimization**
- **`optim.py`** - AdamW + Muon optimizer
  - Single GPU and distributed training support
  - Learning rate scheduling

- **`checkpoint_manager.py`** - Save/load model checkpoints
  - Handles distributed checkpointing
  - Checkpoint metadata and versioning

**Evaluation**
- **`core_eval.py`** - DCLM CORE score evaluation
  - Primary metric for model capability (0.256525 = GPT-2)
  - Used in leaderboard

- **`loss_eval.py`** - Bits-per-byte (BPB) evaluation
  - Vocabulary-size-invariant loss metric
  - Better for comparing models of different sizes

**Inference & Chat**
- **`engine.py`** - Efficient model inference with KV cache
  - Autoregressive text generation
  - Memory-efficient attention caching

- **`execution.py`** - Python code execution as tool
  - Allows model to execute Python code as an extension

**Utilities**
- **`common.py`** - Misc utilities and quality-of-life helpers
  - `COMPUTE_DTYPE` - Global precision management (bf16/fp32/fp16)
  - Device detection, path utilities, etc.

- **`report.py`** - Report writing utilities
  - Training progress summaries

**Utilities**
- **`flash_attention.py`** - Flash attention optimizations (optional)

- **`fp8.py`** - FP8 quantization support
  - Memory-efficient mixed precision training

- **`ui.html`** - Web UI frontend (HTML/CSS/JS)
  - ChatGPT-like interface for `scripts.chat_web`

- **`logo.svg`** - nanochat logo

### `scripts/` - Training & Inference Pipelines

**Pretraining**
- **`base_train.py`** - Base model pretraining script
  - Main entry point: `python -m scripts.base_train`
  - Training loop, loss computation, distributed training
  - Uses DDP for multi-GPU training

- **`base_eval.py`** - Base model evaluation
  - Compute CORE score, BPB, sample generations
  - Run after pretraining to assess quality

**Chat/Fine-tuning**
- **`chat_sft.py`** - Supervised fine-tuning (SFT)
  - Train chat model on conversation data
  - Used after pretraining to add instruction-following

- **`chat_rl.py`** - Reinforcement learning
  - PPO-style training for chat model
  - Reward signals from evaluation tasks

- **`chat_eval.py`** - Chat model evaluation
  - Benchmark on ARC, GSM8K, MMLU, HumanEval, etc.
  - Task-specific metrics

**Inference & Chat**
- **`chat_cli.py`** - Chat via command line
  - Entry point: `python -m scripts.chat_cli -p "prompt here"`
  - Interactive prompt-response loop

- **`chat_web.py`** - Chat via web UI
  - Entry point: `python -m scripts.chat_web`
  - Serves web UI on `localhost:8000`

**Tokenizer**
- **`tok_train.py`** - Train BPE tokenizer
  - Learn vocabulary from corpus

- **`tok_eval.py`** - Evaluate tokenizer compression

### `tasks/` - Evaluation Tasks

Collection of benchmark tasks used in `chat_eval.py` and `chat_rl.py`:

- **`common.py`** - Task infrastructure
  - `Task` - Base class for evaluation tasks
  - `TaskMixture` - Combine multiple tasks
  - `TaskSequence` - Sequential task evaluation

- **`arc.py`** - ARC (AI2 Reasoning Challenge)
  - Multiple-choice science questions (Grade 3-12)

- **`gsm8k.py`** - GSM8K (Grade School Math)
  - Grade school math word problems

- **`mmlu.py`** - MMLU (Massive Multitask Language Understanding)
  - Multiple-choice questions across 57 subjects

- **`humaneval.py`** - Python coding tasks
  - Simple Python coding problems

- **`spellingbee.py`** - Spelling & letter counting
  - Educational task: count specific letters in words

- **`smoltalk.py`** - SmolTalk dataset
  - Conversational data aggregation from HuggingFace

- **`customjson.py`** - Custom JSON tasks
  - Load arbitrary conversation data from JSONL format

### `runs/` - Training Scripts & Configurations

Ready-to-run shell scripts for different scenarios:

- **`speedrun.sh`** - Train ~$100 GPT-2 (reference baseline)
  - Takes ~3 hours on 8XH100
  - Full pipeline: tokenizer → pretrain → SFT → eval → chat
  - Used for leaderboard entries

- **`miniseries.sh`** - Train compute-optimal models at various depths
  - Sweeps depth=12,16,20,24,26,32 etc.
  - Useful for scaling laws research

- **`scaling_laws.sh`** - Scientific scaling laws experiments
  - Systematic evaluation of model/data scaling
  - Generates plots and analysis

- **`runcpu.sh`** - CPU/Apple Silicon example
  - Much smaller model for fast prototyping (~20 min)
  - Not intended for strong results

### `dev/` - Development & Analysis Tools

- **`gen_synthetic_data.py`** - Generate synthetic conversation data
  - For infusing model identity/personality
  - Used in SFT stage

- **`repackage_data_reference.py`** - Pretraining data shard generation
  - Utility for preprocessing training corpora

- **`LEADERBOARD.md`** - Leaderboard documentation
  - How to interpret metrics
  - How to submit entries

- **`LOG.md`** - Development log and notes

- **`estimate_gpt3_core.ipynb`** - Jupyter notebook
  - Analysis of GPT-3 CORE scores

- **`scaling_analysis.ipynb`** - Jupyter notebook
  - Analysis of scaling laws results

- **`nanochat.png`** - Project logo/image

- **`generate_logo.html`** - Logo generation script

### `tests/` - Unit Tests

- **`test_engine.py`** - Tests for inference engine
  - KV cache correctness
  - Generation outputs

- **`test_attention_fallback.py`** - Attention implementation tests

---

## Training Pipeline Flow

### 1. **Tokenizer Training** (`scripts/tok_train.py`)
- Learn BPE vocabulary from raw text
- Creates token encoding for language

### 2. **Pretraining** (`scripts/base_train.py`)
- Next-token prediction on large corpus (Fineweb, ClimbMix, etc.)
- Base model learns general language knowledge
- Outputs: checkpoint with trained weights
- Metric: `val_bpb` (bits per byte), `core_metric` (DCLM CORE)

### 3. **Evaluation** (`scripts/base_eval.py`)
- Assess base model quality
- Compute CORE score (primary metric for speedrun)
- Generate sample outputs

### 4. **Supervised Fine-Tuning (SFT)** (`scripts/chat_sft.py`)
- Fine-tune on conversation data
- Teaches instruction-following & chat format
- Uses synthetic data + curated datasets

### 5. **Reinforcement Learning** (`scripts/chat_rl.py`)
- Optional: improve model via reward signals
- PPO-based training

### 6. **Chat Evaluation** (`scripts/chat_eval.py`)
- Benchmark on standardized tasks (ARC, GSM8K, MMLU, etc.)
- Measure instruction-following capability

### 7. **Inference** (`scripts/chat_cli.py`, `scripts/chat_web.py`)
- Use trained model for real conversations
- CLI or web UI interface

---

## Key Design Principles

### Simplicity via Single Dial (`--depth`)
All hyperparameters are auto-calculated from depth:
```
--depth=12  →  GPT-1 sized (~125M params)
--depth=20  →  Intermediate (~340M params)
--depth=26  →  GPT-2 sized (~1.6B params)
```

Models are automatically **compute-optimal** at their scale.

### Precision Management
Via `COMPUTE_DTYPE` environment variable (not `torch.amp.autocast`):

| Hardware | Default | Override |
|----------|---------|----------|
| H100/A100 (SM 80+) | `bfloat16` | `NANOCHAT_DTYPE=float32` |
| V100/T4 (SM <80) | `float32` | `NANOCHAT_DTYPE=float16` |
| CPU/MPS | `float32` | - |

Model weights stored in fp32 (optimizer precision), embeddings in `COMPUTE_DTYPE`.

### Distributed Training
- DDP (Distributed Data Parallel) via `torchrun`
- Single GPU works too: omit `torchrun`, uses gradient accumulation
- Automatic GPU detection and VRAM optimization

---

## Important Metrics

### CORE Score
- DCLM CORE metric from DCLM paper
- Vocabulary-size-invariant, comparable across model sizes
- **GPT-2 baseline**: 0.256525
- **Goal**: Train model exceeding GPT-2 CORE in <3 hours

### Bits Per Byte (BPB)
- `val_bpb` = validation loss in bits-per-byte
- Independent of tokenizer size (unlike perplexity)
- Lower is better

### Model FLOPs Utilization (MFU)
- `train/mfu` = actual FLOPs / theoretical peak FLOPs
- Indicates training efficiency
- Target: >50% for efficient training

### Tokens Per Second
- `train/tok_per_sec` = training throughput
- Monitor for performance regression

---

## Current Leaderboard Status (as of Mar 2026)

| Rank | Time | CORE | Description | Date |
|------|------|------|-------------|------|
| 0 | 168 hrs | 0.2565 | Original GPT-2 (2019) | - |
| 1 | 3.04 hrs | 0.2585 | d24 baseline | Jan 29 |
| 5 | **1.65 hrs** | 0.2626 | Autoresearch round 2 | Mar 14 |

**Goal**: Continuously improve wall-clock time to exceed GPT-2 CORE score.

---

## Getting Help

- **DeepWiki**: https://deepwiki.com/karpathy/nanochat
- **Discussions**: https://github.com/karpathy/nanochat/discussions
- **Discord**: #nanochat channel in Karpathy's server

---

## AI Contribution Policy

When submitting PRs: disclose any substantial LLM contributions and indicate parts you don't fully understand. This ensures code maintainability.

---

## Quick Reference Commands

```bash
# Train GPT-2 (full pipeline)
bash runs/speedrun.sh

# Quick experiment (5 min)
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_train --depth=12 --run="d12"

# Evaluate trained model
python -m scripts.base_eval --model_path=checkpoints/latest

# Fine-tune on chat data
python -m scripts.chat_sft --model_path=checkpoints/latest

# Chat via CLI
python -m scripts.chat_cli -p "Hello"

# Chat via web UI
python -m scripts.chat_web  # → http://localhost:8000

# Run with custom precision
NANOCHAT_DTYPE=float32 python -m scripts.base_train

# Single GPU (gradient accumulation)
python -m scripts.base_train --depth=12  # (no torchrun)
```

---

## Project License

MIT License - See LICENSE file
