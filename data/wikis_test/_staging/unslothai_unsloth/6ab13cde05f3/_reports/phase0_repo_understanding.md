# Phase 0: Repository Understanding Report

## Summary
- Files explored: 116/116
- Completion: 100%

## Repository Overview

**Unsloth** is a high-performance fine-tuning library for Large Language Models that achieves 2x faster training and 70% less memory usage through:
- Custom Triton kernels for attention, normalization, and activations
- Fused LoRA operations with quantization support (4-bit, FP8)
- Padding-free sequence packing
- Automatic attention backend selection (Flash Attention, xformers, SDPA)

## Key Discoveries

### Main Entry Points
| File | Purpose |
|------|---------|
| `unsloth/__init__.py` | Package initialization - must be imported first to enable optimizations |
| `unsloth/models/loader.py` | `FastLanguageModel.from_pretrained()` - unified model loading API |
| `unsloth/trainer.py` | `UnslothTrainer` - extends HuggingFace SFTTrainer with optimizations |
| `unsloth/save.py` | Model saving: LoRA, merged 16bit, GGUF, Ollama formats |

### Core Modules Identified

#### 1. Model Implementations (`unsloth/models/`)
- **llama.py** (3400 lines): Foundation module with reusable attention patterns
- **gemma.py/gemma2.py**: Gemma support with GeGLU and softcapping
- **mistral.py**: Sliding window attention
- **qwen2.py/qwen3.py/qwen3_moe.py**: Qwen family with QK normalization
- **cohere.py/granite.py**: Alternative architectures
- **falcon_h1.py**: Hybrid attention-SSM model
- **vision.py**: Vision-language model framework (Qwen-VL, LLaVA)

#### 2. Triton Kernels (`unsloth/kernels/`)
- **cross_entropy_loss.py**: Chunked cross-entropy for large vocabularies
- **fast_lora.py**: Fused LoRA for MLP/attention (LoRA_MLP, LoRA_QKV, LoRA_W)
- **rms_layernorm.py**: RMS normalization with Gemma +1 variant
- **rope_embedding.py**: Rotary position embeddings
- **swiglu.py/geglu.py**: Gated activation functions
- **fp8.py**: FP8 quantization with FBGEMM/TorchAO backends
- **flex_attention.py**: Softcapped attention for Gemma2

#### 3. MoE (Mixture of Experts) Kernels (`unsloth/kernels/moe/`)
- Grouped GEMM implementations with Triton
- Reference implementations for Llama4 and Qwen3 MoE
- Autotuning infrastructure for kernel optimization
- Comprehensive test suite for correctness validation

#### 4. Utilities and Infrastructure
- **chat_templates.py**: Jinja2 templates for 50+ model formats
- **tokenizer_utils.py**: Fast tokenizer conversion and validation
- **ollama_template_mappers.py**: Ollama deployment support
- **registry/**: Model registry with search API

### Architecture Patterns Observed

1. **Inheritance Pattern**: All model implementations extend from `llama.py` patterns
2. **Monkey-Patching**: At import time, HuggingFace/TRL/PEFT libraries are patched
3. **Backend Abstraction**: Attention dispatch selects optimal backend automatically
4. **Quantization Layers**: Support for 4-bit (bitsandbytes), FP8 (TorchAO/FBGEMM)
5. **Device Agnostic**: Works on NVIDIA (CUDA), AMD (HIP), and Intel (XPU) GPUs

### File Statistics

| Category | Files | Lines | Key Files |
|----------|-------|-------|-----------|
| Core Package | 76 | ~35,000 | loader.py, llama.py, save.py |
| Kernel Code | 11 | ~5,000 | fast_lora.py, cross_entropy_loss.py |
| MoE Kernels | 21 | ~8,500 | interface.py, forward.py, backward.py |
| Test Files | 38 | ~10,000 | test_unsloth_save.py |
| Scripts | 3 | ~650 | unsloth-cli.py |

## Recommendations for Next Phase

### Suggested Workflows to Document

1. **QLoRA Fine-tuning Workflow**
   - Entry: `FastLanguageModel.from_pretrained()` with `load_in_4bit=True`
   - Training: `UnslothTrainer` or `SFTTrainer`
   - Save: `model.save_pretrained_merged()` or `model.push_to_hub()`
   - Key files: `loader.py`, `trainer.py`, `save.py`

2. **GGUF Export Workflow**
   - Train model with any quantization
   - Export to GGUF with `save_to_gguf()`
   - Optional Ollama integration
   - Key files: `save.py`, `ollama_template_mappers.py`

3. **Vision-Language Model Workflow**
   - Load with `FastVisionModel`
   - Apply LoRA to vision + language components
   - Train on image-text pairs
   - Key files: `vision.py`, `loader.py`

4. **Reinforcement Learning Workflow**
   - GRPO/PPO training with TRL integration
   - vLLM backend for efficient sampling
   - Key files: `rl.py`, `rl_replacements.py`

### Key APIs to Trace

1. `FastLanguageModel.from_pretrained()` → model loading pipeline
2. `model.get_peft_model()` → LoRA adapter creation
3. `UnslothTrainer.train()` → training loop with optimizations
4. `model.save_pretrained_merged()` → weight merging and saving
5. `fast_cross_entropy_loss()` → optimized loss computation

### Important Files for Anchoring Phase

**Highest Priority:**
- `unsloth/__init__.py` - Import order and patch initialization
- `unsloth/models/loader.py` - Model loading entry point
- `unsloth/models/llama.py` - Core attention implementation
- `unsloth/kernels/fast_lora.py` - LoRA optimization

**Medium Priority:**
- `unsloth/trainer.py` - Training integration
- `unsloth/save.py` - Model export
- `unsloth/kernels/cross_entropy_loss.py` - Loss optimization
- `unsloth/chat_templates.py` - Prompt formatting

**Supporting:**
- `unsloth/models/_utils.py` - Low-level utilities
- `unsloth/kernels/utils.py` - Kernel infrastructure
- `unsloth/registry/registry.py` - Model discovery

## Technical Highlights

### Performance Optimizations
- **Fused LoRA**: Combines gate/up/down projections into single autograd functions
- **Chunked Cross-Entropy**: Handles vocabularies >65K without OOM
- **Padding-Free Training**: Eliminates padding tokens from attention
- **Gradient Checkpointing**: Memory-efficient backpropagation
- **torch.compile Integration**: Optional compiler acceleration

### Supported Model Families
- Meta: Llama 3.x, Llama 4 (stub)
- Google: Gemma, Gemma 2
- Mistral AI: Mistral, Mistral Nemo
- Alibaba: Qwen 2.x, Qwen 3 (including MoE)
- IBM: Granite
- Cohere: Command-R
- TII: FalconH1 (hybrid attention-SSM)
- Vision: Qwen-VL, LLaVA-style models

### Deployment Targets
- HuggingFace Hub (native format)
- GGUF (llama.cpp, Ollama)
- TorchAO quantized checkpoints
- vLLM-compatible outputs
