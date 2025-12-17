# Phase 0: Repository Understanding Report

## Summary
- **Files explored:** 200/200
- **Completion:** 100%

## Repository Overview

PEFT (Parameter-Efficient Fine-Tuning) is a HuggingFace library that provides state-of-the-art parameter-efficient fine-tuning methods for large language models. The library implements 20+ adapter methods that enable fine-tuning of massive models with minimal trainable parameters.

### Code Statistics
- **Total Python Files:** 200
- **Total Lines of Code:** 78,061
- **Core Library:** ~50,000 lines in `src/peft/`
- **Tests:** ~35,000 lines in `tests/`

## Key Discoveries

### Main Entry Points
1. **`src/peft/__init__.py`** - Primary package entry point exposing all public APIs
2. **`src/peft/peft_model.py`** (3,387 lines) - Core model wrapper classes:
   - `PeftModel` - Base wrapper for all PEFT adapters
   - `PeftModelForCausalLM`, `PeftModelForSeq2SeqLM`, `PeftModelForSequenceClassification`, etc.
3. **`src/peft/mapping_func.py`** - `get_peft_model()` factory function - THE primary API
4. **`src/peft/auto.py`** - `AutoPeftModel` classes for automatic loading

### Core Modules Identified

#### 1. Tuners (Adapter Implementations)
The library implements 20+ adapter methods organized in `src/peft/tuners/`:

**Low-Rank Methods:**
- **LoRA** (`lora/`) - Original low-rank adaptation, 18 files, most complete implementation
- **AdaLoRA** (`adalora/`) - Adaptive rank allocation with SVD decomposition
- **VeRA** (`vera/`) - Vector-based random adaptation (10-100x fewer params than LoRA)
- **VBLoRA** (`vblora/`) - Vector bank LoRA with top-k selection
- **RandLoRA** (`randlora/`) - Random shared basis projections
- **GraLoRA** (`gralora/`) - Block-structured gradient low-rank

**Orthogonal/Rotation Methods:**
- **OFT** (`oft/`) - Orthogonal Fine-Tuning with Cayley parametrization
- **BOFT** (`boft/`) - Butterfly-factorized orthogonal transforms
- **HRA** (`hra/`) - Householder reflection adaptation
- **MISS** (`miss/`) - Mixture of Sharded Squares (replaces deprecated BONE)
- **RoAd** (`road/`) - 2D rotation adaptation

**Other Methods:**
- **IA3** (`ia3/`) - Infused Adapter by Inhibiting and Amplifying activations
- **LoHa** (`loha/`) - Low-rank Hadamard product (LyCORIS family)
- **Poly** (`poly/`) - Polytropon multi-task with skill routing
- **FourierFT** (`fourierft/`) - Frequency-domain fine-tuning
- **C3A** (`c3a/`) - Block circulant convolution
- **SHiRA** (`shira/`) - Sparse High-Rank Adaptation

**Prompt-Based Methods:**
- **Adaption Prompt** (`adaption_prompt/`) - LLaMA-Adapter style
- **Multitask Prompt Tuning** (`multitask_prompt_tuning/`) - Factorized prompts
- **CPT** (`cpt/`) - Context-aware Prompt Tuning

#### 2. Utilities (`src/peft/utils/`)
- **`save_and_load.py`** - Adapter serialization/deserialization
- **`hotswap.py`** - Rapid adapter switching without model reload
- **`merge_utils.py`** - Multi-adapter merging techniques (TIES, DARE, linear)
- **`integrations.py`** - DeepSpeed, bitsandbytes, GPTQ integration
- **`loftq_utils.py`** - LoftQ quantization-aware initialization
- **`constants.py`** - Model architecture mappings and defaults

#### 3. Optimizers (`src/peft/optimizers/`)
- **LoRA-FA** - Frozen-A training for memory efficiency
- **LoRA+** - Differential learning rates for lora_A/lora_B

### Architecture Patterns Observed

1. **Config → Model → Layer Pattern**
   Each tuner follows a consistent structure:
   - `config.py` - Dataclass with parameters
   - `model.py` - Orchestration and injection logic
   - `layer.py` - Actual forward pass implementation

2. **Quantization Backend Support**
   Most methods support multiple quantization backends:
   - BitsAndBytes (4-bit/8-bit QLoRA)
   - GPTQ
   - AWQ
   - AQLM
   - EETQ
   - HQQ
   - Intel Neural Compressor
   - TorchAO

3. **Base Class Hierarchy**
   - `BaseTuner` / `BaseTunerLayer` in `tuners_utils.py` (2,041 lines)
   - `LycorisConfig` / `LycorisLayer` for factorization methods
   - All methods inherit from these base classes

4. **Lazy Imports**
   Heavy use of lazy imports to minimize startup time and optional dependencies

### Test Coverage
The test suite is comprehensive with 46 test files covering:
- All adapter methods individually
- GPU operations and quantization
- Model architectures (decoder, encoder-decoder, vision)
- Integrations (DeepSpeed, FSDP, Megatron)
- Edge cases and error handling

### Supporting Tools
- **method_comparison/** - Gradio app for visualizing PEFT method comparisons
- **scripts/** - CI utilities, memory profiling, checkpoint conversion

## Recommendations for Next Phase

### Suggested Workflows to Document
1. **Basic LoRA Fine-Tuning** - Load model → Apply LoRA → Train → Save/Load
2. **QLoRA 4-bit Training** - Setup bitsandbytes → Configure → Train
3. **Multi-Adapter Management** - Add adapters → Switch → Merge
4. **Hot-Swapping Adapters** - Deploy → Swap without reload
5. **Custom Model Integration** - Target modules → Configuration

### Key APIs to Trace
1. `get_peft_model()` - Primary entry point
2. `PeftModel.from_pretrained()` - Loading adapters
3. `model.merge_and_unload()` - Merging adapters to base
4. `hotswap_adapter()` - Zero-downtime switching
5. `merge_adapters()` - Combining multiple adapters

### Important Files for Anchoring Phase
| File | Importance | Reason |
|------|------------|--------|
| `src/peft/peft_model.py` | Critical | Core wrapper, all operations flow through here |
| `src/peft/tuners/lora/model.py` | High | Most complete tuner implementation |
| `src/peft/tuners/tuners_utils.py` | High | Base classes all tuners inherit |
| `src/peft/utils/save_and_load.py` | High | Serialization is key for deployment |
| `src/peft/mapping_func.py` | High | Factory function users call |

### Architectural Insights for Documentation
1. **Plugin Architecture** - Easy to add new tuner methods
2. **Quantization as First-Class** - Not an afterthought
3. **Multi-Adapter Support** - Core design principle
4. **Transformers Integration** - Deep HuggingFace ecosystem ties
5. **Memory Optimization** - Throughout (gradient checkpointing, offloading)

## File Categories Summary

| Category | Count | Key Files |
|----------|-------|-----------|
| Core PEFT | 14 | peft_model.py, mapping_func.py, auto.py |
| LoRA Tuner | 18 | layer.py, model.py, config.py, variants.py |
| Other Tuners | 95 | 19 different tuner modules |
| Utilities | 11 | save_and_load.py, hotswap.py, merge_utils.py |
| Optimizers | 3 | lorafa.py, loraplus.py |
| Tests | 46 | test_custom_models.py, test_gpu_examples.py |
| Scripts | 6 | train_memory.py, ci_clean_cache.py |
| Method Comparison | 5 | app.py, processing.py |
| Setup | 1 | setup.py |

## Conclusion

PEFT is a well-structured, highly modular library with consistent patterns across all adapter implementations. The codebase is mature with extensive testing and supports a wide range of quantization backends. The documentation phase should focus on:

1. Getting started workflows for common use cases
2. Deep-dives into LoRA and its variants (most popular)
3. Quantization setup guides (QLoRA is very popular)
4. Multi-adapter and hot-swapping for production use
5. Architecture overview for contributors
