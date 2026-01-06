# Phase 0: Repository Understanding Report

## Summary
- **Files explored:** 200/200
- **Completion:** 100%
- **Total lines of code:** 78,061

## Repository Overview

PEFT (Parameter-Efficient Fine-Tuning) is HuggingFace's library for efficient fine-tuning of large language models. It implements 25+ different adapter methods that modify only a small subset of model parameters while achieving competitive performance with full fine-tuning.

## Key Discoveries

### Main Entry Points
1. **`src/peft/__init__.py`** - Package entry point exporting all public APIs
2. **`src/peft/peft_model.py`** - Core `PeftModel` wrapper class (3,387 lines)
3. **`src/peft/mapping_func.py`** - `get_peft_model()` factory function
4. **`src/peft/auto.py`** - `AutoPeftModel` classes for easy loading

### Core Modules Identified

#### Configuration System
- `src/peft/config.py` - Base `PeftConfig` and `PromptLearningConfig` classes
- Each tuner has its own config class inheriting from these bases

#### Tuner Methods (25+ implementations)

**Low-Rank Adaptation Family:**
- **LoRA** (`tuners/lora/`) - 18 files, core low-rank adaptation
- **AdaLoRA** (`tuners/adalora/`) - Adaptive rank allocation via SVD
- **DoRA** (in lora/variants.py) - Weight-decomposed LoRA
- **GraLoRA** (`tuners/gralora/`) - Block-wise low-rank decomposition
- **VeRA** (`tuners/vera/`) - Vector-based random matrix adaptation
- **VBLoRA** (`tuners/vblora/`) - Vector bank with top-K selection
- **RandLoRA** (`tuners/randlora/`) - Shared frozen random projections

**Orthogonal Methods:**
- **OFT** (`tuners/oft/`) - Orthogonal fine-tuning with Cayley transform
- **BOFT** (`tuners/boft/`) - Butterfly orthogonal factorization
- **HRA** (`tuners/hra/`) - Householder reflection adaptation
- **MiSS** (`tuners/miss/`) - Mixture of subspaces (replaces BONE)

**Prompt-Based Methods:**
- **Prompt Tuning** (`tuners/prompt_tuning/`) - Direct soft prompt learning
- **P-Tuning** (`tuners/p_tuning/`) - Neural prompt encoder (MLP/LSTM)
- **Prefix Tuning** (`tuners/prefix_tuning/`) - Prefix key-value injection
- **Adaption Prompt** (`tuners/adaption_prompt/`) - LLaMA-Adapter style

**Other Methods:**
- **IA3** (`tuners/ia3/`) - Per-dimension scaling vectors
- **LoHa** (`tuners/loha/`) - Hadamard product factorization
- **LoKr** (`tuners/lokr/`) - Kronecker product factorization
- **Poly** (`tuners/poly/`) - Polytropon multi-task routing
- **X-LoRA** (`tuners/xlora/`) - Mixture-of-LoRA-experts
- **SHiRA** (`tuners/shira/`) - Sparse high-rank adaptation
- **RoAd** (`tuners/road/`) - Rotation-based adaptation
- **FourierFT** (`tuners/fourierft/`) - Fourier domain fine-tuning
- **C3A** (`tuners/c3a/`) - Circular convolution adapter
- **CPT** (`tuners/cpt/`) - Context-aware prompt tuning
- **LN Tuning** (`tuners/ln_tuning/`) - LayerNorm-only tuning
- **Trainable Tokens** (`tuners/trainable_tokens/`) - Selective token training

### Architecture Patterns Observed

1. **Consistent Module Structure:** Each tuner follows the pattern:
   - `__init__.py` - Exports and method registration
   - `config.py` - Configuration dataclass
   - `model.py` - Tuner model class (inherits BaseTuner)
   - `layer.py` - Adapter layer implementations

2. **Registry Pattern:** `register_peft_method()` in `utils/peft_types.py` populates global mappings that enable dynamic dispatch

3. **Quantization Support:** Extensive support for 8 quantization backends:
   - bitsandbytes (8-bit, 4-bit)
   - GPTQ (auto-gptq, gptqmodel)
   - AWQ
   - AQLM
   - EETQ
   - HQQ
   - Intel Neural Compressor (FP8)
   - TorchAO

4. **Mixed Adapter Support:** `PeftMixedModel` and `tuners/mixed/` enable combining different adapter types

5. **LyCORIS Compatibility:** `tuners/lycoris_utils.py` provides base class for LoHa/LoKr (Stable Diffusion ecosystem)

### Utility Infrastructure

- **`utils/save_and_load.py`** - Adapter serialization with smart embedding handling
- **`utils/hotswap.py`** - Runtime LoRA swapping without recompilation
- **`utils/merge_utils.py`** - 6 adapter merging algorithms (TIES, DARE, etc.)
- **`utils/loftq_utils.py`** - LoftQ quantization-aware initialization
- **`utils/integrations.py`** - Compatibility with transformers, accelerate, deepspeed

### Optimizer Enhancements
- **LoRA+** (`optimizers/loraplus.py`) - Different learning rates for A/B matrices
- **LoRA-FA** (`optimizers/lorafa.py`) - Frozen-A optimization with projected gradients

### Test Coverage
- 46 test files covering all major functionality
- Comprehensive GPU tests (`test_common_gpu.py`, `test_gpu_examples.py`)
- Architecture-specific tests (decoder, encoder-decoder, vision models)
- Integration tests for diffusers, quantization libraries

## Recommendations for Next Phase

### Suggested Workflows to Document

1. **Basic LoRA Fine-Tuning**
   - Entry: `get_peft_model()` → LoRA config → train → save/load
   - Key files: `mapping_func.py`, `lora/config.py`, `lora/model.py`

2. **Quantized Training (QLoRA)**
   - Entry: bitsandbytes 4-bit model + LoRA
   - Key files: `lora/bnb.py`, `utils/integrations.py`

3. **Adapter Merging**
   - Entry: Multiple trained adapters → merge algorithms
   - Key files: `utils/merge_utils.py`, `peft_model.py` (merge methods)

4. **Production Inference with Hotswap**
   - Entry: torch.compile + adapter swapping
   - Key files: `utils/hotswap.py`

5. **Multi-Adapter Inference**
   - Entry: `PeftMixedModel` with multiple adapter types
   - Key files: `mixed_model.py`, `tuners/mixed/`

### Key APIs to Trace

1. `get_peft_model()` - Primary entry point
2. `PeftModel.from_pretrained()` - Loading saved adapters
3. `inject_adapter_in_model()` - Low-level injection (for integrations)
4. `merge_and_unload()` - Converting PEFT model back to base model
5. `add_adapter()` / `set_adapter()` - Multi-adapter management

### Important Files for Anchoring Phase

**Core (must understand deeply):**
- `src/peft/peft_model.py` - All model operations
- `src/peft/tuners/lora/layer.py` - Fundamental adapter mechanics
- `src/peft/tuners/tuners_utils.py` - Base classes and utilities

**For Quantization Support:**
- `src/peft/tuners/lora/bnb.py` - bitsandbytes integration
- `src/peft/utils/integrations.py` - Backend detection

**For Production:**
- `src/peft/utils/hotswap.py` - Runtime adapter swapping
- `src/peft/utils/save_and_load.py` - Serialization

## Method Comparison Summary

| Method | Params | Best For |
|--------|--------|----------|
| LoRA | 0.1-1% | General-purpose, well-understood |
| DoRA | ~LoRA | Better than LoRA in many cases |
| AdaLoRA | Dynamic | When optimal rank is unknown |
| IA3 | <<0.1% | Extreme parameter efficiency |
| Prompt Tuning | Virtual tokens | Few-shot learning |
| OFT/BOFT | ~LoRA | Activation norm preservation |
| VeRA | <<0.1% | Shared random matrices |
| X-LoRA | Multiple LoRAs | Dynamic multi-task |
| SHiRA | ~LoRA | Full-rank sparse adaptation |
