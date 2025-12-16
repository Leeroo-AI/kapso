# Phase 0: Repository Understanding Report

## Summary
- **Files explored:** 116/116
- **Completion:** 100%
- **Repository:** unslothai/unsloth
- **Total lines of code:** 50,481

## Key Discoveries

### Main Entry Points
1. **`unsloth/__init__.py`** - Critical entry point that MUST be imported before other ML libraries. Sets up monkey-patches and optimizations.
2. **`unsloth/models/loader.py`** - User-facing API with `FastLanguageModel`, `FastVisionModel`, `FastTextModel` classes.
3. **`unsloth-cli.py`** - Command-line interface for training without Python code.

### Core Modules Identified

#### Model Loading & Optimization (`unsloth/models/`)
- **loader.py** (1262 lines) - Main orchestrator for model loading with quantization support
- **llama.py** (3400 lines) - Reference implementation defining optimization patterns (attention, MLP, RMSNorm)
- **_utils.py** (2356 lines) - Shared utilities for patching, version checking, device detection
- Architecture-specific files: gemma.py, gemma2.py, mistral.py, qwen2.py, qwen3.py, cohere.py, granite.py, falcon_h1.py, vision.py

#### Optimized Kernels (`unsloth/kernels/`)
- **Triton-based kernels** for GPU acceleration:
  - `cross_entropy_loss.py` - Chunked loss for large vocabularies
  - `fast_lora.py` - Fused LoRA operations (QKV, MLP, output)
  - `rms_layernorm.py` - RMSNorm for LLaMA models
  - `rope_embedding.py` - Rotary position embeddings
  - `swiglu.py`, `geglu.py` - Activation kernels
  - `fp8.py` - FP8 quantization support

#### MoE Support (`unsloth/kernels/moe/`)
- **Grouped GEMM** implementation for Mixture of Experts
- Reference implementations for Llama4 and Qwen3 MoE architectures
- Autotuning infrastructure for optimal kernel configurations

#### Export & Deployment (`unsloth/save.py`, `unsloth/ollama_template_mappers.py`)
- **save.py** (3068 lines) - Model saving in multiple formats: HuggingFace, GGUF, Ollama
- **ollama_template_mappers.py** (2192 lines) - 40+ chat templates for Ollama deployment
- Support for 20+ GGUF quantization methods (q4_k_m, q5_k_m, q8_0, etc.)

#### Training Infrastructure
- **trainer.py** - Custom `UnslothTrainer` with sample packing and backwards-compatible patches
- **models/rl.py**, **models/rl_replacements.py** - RL training support (GRPO, DPO, PPO, SFT)
- **utils/packing.py** - Sequence packing for efficient batching

#### Model Registry (`unsloth/registry/`)
- Centralized model information management
- Support for 6 model families: DeepSeek, Gemma, Llama, Mistral, Phi, Qwen
- 5 quantization types: BNB, UNSLOTH, GGUF, NONE, BF16

### Architecture Patterns Observed

1. **Monkey-patching approach** - Unsloth replaces transformers/TRL methods at import time with optimized versions
2. **Inheritance hierarchy** - Model files inherit from `FastLlamaModel` and customize architecture-specific differences
3. **Multi-hardware support** - Device detection for CUDA/HIP(AMD)/XPU(Intel) with appropriate fallbacks
4. **Quantization integration** - Seamless handling of 4-bit (BnB), 8-bit, FP8, and 16-bit models
5. **Backwards compatibility** - Extensive patching to work across multiple TRL and transformers versions

## File Distribution

| Category | Count | Lines |
|----------|-------|-------|
| Package (core library) | 76 | ~45,000 |
| Tests | 37 | ~4,800 |
| Scripts/Other | 3 | ~650 |

## Recommendations for Next Phase

### Suggested Workflows to Document
1. **Model Loading Workflow** - `FastLanguageModel.from_pretrained()` → quantization → LoRA patching
2. **Training Workflow** - SFTTrainer setup → sample packing → gradient accumulation → checkpointing
3. **Export Workflow** - Model merging → GGUF conversion → Ollama deployment
4. **Vision Model Workflow** - VLM loading → multimodal processing → vLLM integration

### Key APIs to Trace
- `FastLanguageModel.from_pretrained()` - Entry point for all model loading
- `FastLanguageModel.get_peft_model()` - LoRA configuration
- `model.save_pretrained_merged()` - Model export
- `model.save_pretrained_gguf()` - GGUF export

### Important Files for Anchoring Phase
1. **unsloth/models/loader.py** - Main API entry point
2. **unsloth/models/llama.py** - Reference implementation
3. **unsloth/save.py** - Export functionality
4. **unsloth/trainer.py** - Training configuration
5. **unsloth/kernels/utils.py** - Core kernel infrastructure

### Critical Dependencies
- bitsandbytes (4-bit quantization)
- peft (LoRA support)
- trl (training infrastructure)
- triton (GPU kernels)
- huggingface_hub (model distribution)
- llama.cpp (GGUF conversion)

## Notes for Documentation

- Unsloth MUST be imported before transformers/trl/peft for optimizations to apply
- Many models use the same underlying code (Qwen2 reuses Llama code with thin wrapper)
- The codebase prioritizes performance over readability - lots of low-level optimizations
- Test files provide good examples of intended usage patterns
- Chat templates are critical for correct inference - see `chat_templates.py` and `ollama_template_mappers.py`
