# Phase 0: Repository Understanding Report

## Summary
- Files explored: 116/116
- Completion: 100%

## Key Discoveries

### Main Entry Points
1. **`unsloth/__init__.py`** - Bootstrap module that validates import order, applies compatibility patches, detects hardware (CUDA/HIP/XPU), and exports the public API
2. **`unsloth/models/loader.py`** - Main user-facing API with `FastLanguageModel`, `FastModel`, `FastVisionModel` classes for loading and optimizing models
3. **`unsloth/save.py`** - Comprehensive model export system supporting LoRA, merged 16-bit, merged 4-bit, and GGUF formats
4. **`unsloth-cli.py`** - Complete CLI for fine-tuning without writing code

### Core Modules Identified

#### Model Loading & Optimization (`unsloth/models/`)
- **`loader.py`** (1,262 lines) - Main model loading interface with quantization support
- **`llama.py`** (3,400 lines) - Foundation implementation with optimized attention, RoPE, SwiGLU, gradient checkpointing
- **`vision.py`** (1,263 lines) - Contains `FastBaseModel`, the base class for ALL model implementations (despite the name)
- **Model adapters**: `gemma.py`, `gemma2.py`, `mistral.py`, `qwen2.py`, `qwen3.py`, `cohere.py`, `granite.py`, `falcon_h1.py`
- **`mapper.py`** (1,324 lines) - Bidirectional mappings between quantized/full-precision model variants

#### Custom Triton Kernels (`unsloth/kernels/`)
- **Activations**: `swiglu.py`, `geglu.py`
- **Normalization**: `layernorm.py`, `rms_layernorm.py`
- **Position encoding**: `rope_embedding.py`
- **Loss functions**: `cross_entropy_loss.py` (supports 256K+ vocabularies)
- **LoRA fusion**: `fast_lora.py` - Fused LoRA operations for MLP/QKV/output layers
- **Quantization**: `fp8.py` - FP8 support with multiple backends (FBGEMM, TorchAO, Triton)
- **MoE subsystem**: Complete Mixture-of-Experts implementation with grouped GEMM kernels, autotuning, and reference implementations for Llama4 and Qwen3

#### Training Infrastructure
- **`trainer.py`** - Extends TRL's SFTTrainer with padding-free training and sample packing (>2x speedup)
- **`unsloth/models/rl.py`** + **`rl_replacements.py`** - RL training integration (GRPO, PPO, DPO) with vLLM acceleration
- **`unsloth/utils/packing.py`** - Sequence packing for efficient batching

#### Data & Templates
- **`chat_templates.py`** (3,159 lines) - 30+ Jinja2 templates for instruction-tuned models
- **`ollama_template_mappers.py`** (2,192 lines) - 50+ Ollama Modelfile templates
- **`tokenizer_utils.py`** (1,105 lines) - Tokenizer validation and repair utilities
- **`dataprep/synthetic.py`** - Synthetic data generation using vLLM

#### Model Registry (`unsloth/registry/`)
- Centralized model metadata system tracking supported models by organization (Meta, Google, Alibaba, Microsoft, Mistral, DeepSeek)
- Supports multiple quantization types: BNB, UNSLOTH, GGUF, BF16

### Architecture Patterns Observed

1. **Hierarchical Model Design**: `llama.py` serves as the foundation, with other architectures extending it
2. **Monkey-patching for Optimization**: Heavy use of runtime patching to replace HuggingFace/TRL implementations
3. **Multi-backend Support**: Graceful fallbacks across NVIDIA/AMD/Intel GPUs and various library versions
4. **Defensive Programming**: Extensive validation, fixing, and compatibility patches in `import_fixes.py`
5. **Fused Operations**: All kernels fuse operations to minimize memory traffic

### Performance Optimizations
- **Sample packing**: Combines multiple sequences into single batches, eliminating padding waste
- **Fused LoRA kernels**: Combine base weights + adapters in single operations
- **Triton kernels**: Custom implementations for all compute-heavy operations
- **FP8/4-bit quantization**: Multiple quantization schemes for memory efficiency

## Recommendations for Next Phase

### Suggested Workflows to Document
1. **QLoRA Fine-tuning Workflow**: `loader.py` → model selection → LoRA application → training → merging → export
2. **GGUF Export Workflow**: Training → merging → llama.cpp compilation → quantization → Ollama deployment
3. **RL Training Workflow**: GRPO/PPO/DPO setup with vLLM integration
4. **Vision Model Workflow**: Qwen-VL loading → OCR fine-tuning → evaluation
5. **Synthetic Data Generation**: Document chunking → vLLM inference → QA pair generation

### Key APIs to Trace
1. `FastLanguageModel.from_pretrained()` - Model loading with quantization
2. `FastLanguageModel.get_peft_model()` - LoRA adapter application
3. `model.save_pretrained_merged()` - LoRA merging and export
4. `model.push_to_hub_merged()` - Hub deployment
5. `UnslothTrainer` / `SFTTrainer` integration

### Important Files for Anchoring Phase
1. **Entry points**: `__init__.py`, `loader.py`, `save.py`
2. **Core optimizations**: `llama.py`, `fast_lora.py`, `trainer.py`
3. **Templates**: `chat_templates.py`, `ollama_template_mappers.py`
4. **MoE system**: `grouped_gemm/interface.py`, `moe_ops.py`

## Test Coverage Summary

The test suite validates:
- **QLoRA training/merging**: Both HuggingFace baseline and Unsloth implementations
- **Perplexity preservation**: Across 4-bit, 8-bit, 16-bit quantization for multiple model families
- **Hub integration**: Upload/download of merged and sharded models
- **Model types**: Text LLMs, vision-language models, TTS (CSM, Llasa, Orpheus, Whisper)
- **RL training**: GRPO pipeline with AIME benchmark evaluation
- **Infrastructure**: Sequence packing, attention masks, QAT integration

## Repository Statistics
- **Total Python files**: 116
- **Total lines of code**: 50,480
- **Package files**: 77 (core library)
- **Test files**: 37 (comprehensive test coverage)
- **Script files**: 2 (formatting tools)
- **CLI file**: 1 (unsloth-cli.py)
