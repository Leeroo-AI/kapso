# Phase 0: Repository Understanding Report

## Summary
- Files explored: 118/118
- Completion: 100%

## Repository Overview

**Unsloth** is a high-performance library for fine-tuning large language models (LLMs) with significant memory and speed optimizations. The repository contains ~52,000 lines of Python code across 118 files.

### Core Value Proposition
- 2x faster fine-tuning with 70% less memory
- Triton-based GPU kernels for critical operations
- LoRA/QLoRA support with optimized autograd functions
- Multi-model support (LLaMA, Mistral, Gemma, Qwen, Cohere, Granite, Falcon)
- GGUF export for Ollama/llama.cpp deployment

## Key Discoveries

### Main Entry Points
1. **`unsloth/__init__.py`** - Package initialization, environment setup, compatibility checks
2. **`unsloth/models/loader.py`** - `FastLanguageModel.from_pretrained()` - primary API for loading models with quantization
3. **`unsloth/save.py`** - Model saving (merged weights, GGUF, Hub push)
4. **`unsloth/trainer.py`** - `UnslothTrainer` with sample packing support
5. **`unsloth-cli.py`** - Command-line interface for training

### Core Modules Identified

#### Kernel Layer (`unsloth/kernels/`)
- **Triton kernels** for GPU-accelerated operations:
  - `cross_entropy_loss.py` - Chunked cross entropy for large vocabularies (256K+)
  - `fast_lora.py` - Fused LoRA forward/backward for QKV and MLP
  - `rms_layernorm.py` - RMS normalization with model-specific variants
  - `rope_embedding.py` - Rotary position embeddings
  - `swiglu.py`, `geglu.py` - Activation functions
  - `fp8.py` - FP8 quantization support

#### MoE Support (`unsloth/kernels/moe/`)
- **Mixture of Experts** kernels for Qwen3 and Llama4:
  - Grouped GEMM with autotuning
  - Forward/backward Triton kernels
  - Reference implementations for correctness testing

#### Model Architectures (`unsloth/models/`)
- **Base architecture**: `llama.py` (3,452 lines) - foundation for all models
- **Model-specific patches**:
  - `gemma.py`, `gemma2.py` - GEGLU activation, soft capping
  - `mistral.py` - Sliding window attention
  - `qwen2.py`, `qwen3.py`, `qwen3_moe.py` - Qwen family
  - `cohere.py` - LayerNorm (not RMSNorm)
  - `granite.py` - IBM enterprise models
  - `falcon_h1.py` - Hybrid Mamba-Attention
  - `vision.py` - Vision-language models (Qwen2-VL, Llama 3.2 Vision)

#### Training & RL (`unsloth/models/rl.py`, `rl_replacements.py`)
- GRPO, PPO, DPO trainer patches
- Integration with TRL library
- Custom generation methods for RL training

### Architecture Patterns Observed

1. **Monkey Patching**: Unsloth patches HuggingFace transformers at runtime rather than forking
2. **Inheritance Chain**: Model files inherit from LLaMA base, overriding only architecture-specific methods
3. **Triton JIT Compilation**: Kernels use `@triton.jit` with heuristics for dynamic configuration
4. **Autograd Functions**: Custom `torch.autograd.Function` classes for efficient backward passes
5. **Registry Pattern**: Model registry for managing supported model families and quantization types

### Data Preparation (`unsloth/dataprep/`)
- Raw text to training dataset conversion
- Synthetic data generation via vLLM
- Chat template management (20+ model families)

### Testing Infrastructure (`tests/`)
- **Perplexity benchmarks** for merge quality validation
- **OCR benchmarks** for vision models
- **AIME evaluation** for math reasoning
- **Sample packing tests** for training correctness
- **QAT tests** for quantization-aware training

## Recommendations for Next Phase

### Suggested Workflows to Document
1. **Basic Fine-tuning Workflow**: Load model → Prepare data → Train → Save merged
2. **QLoRA Training**: 4-bit loading → LoRA config → Training → Merge & export
3. **RL Training (GRPO)**: SFT pretraining → GRPO alignment → Evaluation
4. **Vision Model Fine-tuning**: VLM loading → Image preprocessing → Training
5. **GGUF Export**: Fine-tuned model → Quantization → GGUF → Ollama deployment

### Key APIs to Trace
1. `FastLanguageModel.from_pretrained()` - Model loading path
2. `FastLanguageModel.get_peft_model()` - LoRA/QLoRA setup
3. `model.save_pretrained_merged()` - Weight merging
4. `model.push_to_hub_merged()` - Hub deployment
5. `model.save_pretrained_gguf()` - GGUF export

### Important Files for Anchoring Phase
1. **`unsloth/models/loader.py`** - Central hub for model loading
2. **`unsloth/models/llama.py`** - Base architecture reference
3. **`unsloth/kernels/fast_lora.py`** - Core optimization layer
4. **`unsloth/save.py`** - All export pathways
5. **`unsloth/trainer.py`** - Training configuration

### Model Family Support Matrix
| Family | Text | Vision | MoE | RL Support |
|--------|------|--------|-----|------------|
| LLaMA | ✅ | ✅ (3.2) | ✅ (4) | ✅ |
| Qwen | ✅ | ✅ (2-VL, 2.5-VL) | ✅ (3) | ✅ |
| Gemma | ✅ | — | — | ✅ |
| Mistral | ✅ | — | — | ✅ |
| Cohere | ✅ | — | — | ✅ |
| Granite | ✅ | — | — | ✅ |
| Falcon | ✅ | — | — | ✅ |
| Phi | ✅ | — | — | ✅ |
| Deepseek | ✅ | — | ✅ | ✅ |

## Metrics

| Category | Count | Lines |
|----------|-------|-------|
| Package files | 77 | ~45,000 |
| Example/script files | 2 | 209 |
| Test files | 38 | ~6,500 |
| CLI | 1 | 473 |
| **Total** | **118** | **51,799** |

---

*Report generated: 2026-01-09*
*Phase 0 Status: COMPLETE*
