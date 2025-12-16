# Phase 4: Enrichment Report

## Summary

Scanned the unslothai_unsloth repository implementation code to extract Environment constraints and Heuristics (tribal knowledge). Created comprehensive wiki pages documenting hardware/software requirements and practical optimization wisdom.

## Environments Created

| Environment | Required By | Description |
|-------------|-------------|-------------|
| unslothai_unsloth_GPU_CUDA_Environment | FastLanguageModel, FastVisionModel, UnslothTrainer, QLoRA_Finetuning, Vision_Model_Finetuning | CUDA 11.8+, PyTorch 2.0+, Triton 3.0+, bitsandbytes for 4-bit QLoRA training |
| unslothai_unsloth_GGUF_Export_Environment | save_to_gguf, Model_Export_GGUF | llama.cpp build tools (cmake, make, git) for GGUF model export |

### Environment Requirements Summary

#### GPU CUDA Environment
- **Hardware:** NVIDIA GPU with CUDA 7.0+ compute capability (V100, RTX 20xx and newer)
- **Alternative Hardware:** AMD GPUs via ROCm/HIP, Intel XPU (PyTorch 2.6+)
- **Python:** 3.9 - 3.13
- **Key Dependencies:**
  - `torch` >= 2.0.0
  - `triton` >= 3.0.0
  - `transformers` >= 4.51.3
  - `bitsandbytes` >= 0.45.5
  - `peft` >= 0.7.1
  - `trl` >= 0.18.2

#### GGUF Export Environment
- **System Packages:** git, cmake >= 3.14, make (legacy), gcc/g++ >= 9.0
- **Purpose:** Compile llama.cpp for model quantization and export
- **Runtime:** CPU-only (GPU not required for export)

## Heuristics Created

| Heuristic | Applies To | Key Insights |
|-----------|------------|--------------|
| unslothai_unsloth_LoRA_Rank_Selection | FastLanguageModel, FastVisionModel, QLoRA_Finetuning, Low_Rank_Adaptation | r=16 default, r=32-64 for complex tasks, lora_dropout=0 optimized |
| unslothai_unsloth_Memory_Management | FastLanguageModel, UnslothTrainer, QLoRA_Finetuning, Gradient_Checkpointing | maximum_memory_usage=0.9, gradient checkpointing modes, sample packing |
| unslothai_unsloth_GGUF_Quantization_Selection | save_to_gguf, Model_Export_GGUF, GGUF_Model_Quantization | q4_k_m default (best quality/size), q8_0 for quality, q2_k for size |
| unslothai_unsloth_Learning_Rate_Guidelines | UnslothTrainer, QLoRA_Finetuning, Supervised_Fine_Tuning | lr=2e-4 default for QLoRA, embedding_lr=5e-5 for embeddings |
| unslothai_unsloth_Dtype_Selection | FastLanguageModel, FastVisionModel, QLoRA_Finetuning, QLoRA_4bit_Quantization | bfloat16 on Ampere+, float16 on older GPUs, float32 for Gemma3/GPT-OSS |

### Heuristics Detail

#### LoRA Rank Selection
- Default `r=16` is optimal for most instruction-following tasks
- Simple tasks: `r=8` or lower
- Complex tasks (math, code, reasoning): `r=32-64`
- Very large models (70B+): `r=64-128`
- **Unsloth optimization:** `lora_dropout=0` (differs from standard recommendation)

#### Memory Management
- `maximum_memory_usage=0.9` is safe default
- `use_gradient_checkpointing="unsloth"` reduces VRAM by ~30-50%
- Sample packing (`packing=True`) reduces padding waste
- Padding-free training auto-enabled for compatible models
- **Blocklist:** gemma2, gpt_oss don't support padding-free

#### GGUF Quantization Selection
- **q4_k_m:** Best default choice, good quality/size balance
- **q5_k_m:** Slightly better quality than q4_k_m
- **q8_0:** Nearly lossless, 2x size of q4_k_m
- **f16/bf16:** Full precision, fastest conversion
- Shortcuts: `"quantized"` → q4_k_m, `"fast_quantized"` → q8_0

#### Learning Rate Guidelines
- Standard QLoRA: `lr=2e-4`
- Embedding learning rate: `lr=5e-5` (10-20x lower)
- Larger models need lower LR (70B: `lr=5e-5` to `1e-4`)
- Use cosine scheduler for longer training

#### Dtype Selection
- Auto-detection: bfloat16 if supported, else float16
- **Force float32:** Gemma3, Gemma3n, GPT-OSS (stability issues)
- Environment override: `UNSLOTH_FORCE_FLOAT32=1`

## Links Added

### Environment Links
- `_EnvironmentIndex.md`: 2 environment entries with 7 connection references
- `_ImplementationIndex.md`: Added Env links to 4 implementations
- `_WorkflowIndex.md`: Added Env links to 3 workflows

### Heuristic Links
- `_HeuristicIndex.md`: 5 heuristic entries with 17 connection references
- `_ImplementationIndex.md`: Added Heuristic links to 5 implementations
- `_WorkflowIndex.md`: Added Heuristic links to 2 workflows

## Files Written

```
environments/
├── unslothai_unsloth_GPU_CUDA_Environment.md
└── unslothai_unsloth_GGUF_Export_Environment.md

heuristics/
├── unslothai_unsloth_LoRA_Rank_Selection.md
├── unslothai_unsloth_Memory_Management.md
├── unslothai_unsloth_GGUF_Quantization_Selection.md
├── unslothai_unsloth_Learning_Rate_Guidelines.md
└── unslothai_unsloth_Dtype_Selection.md
```

## Index Updates Made

- ✅ Updated `_EnvironmentIndex.md` with 2 environment pages
- ✅ Updated `_HeuristicIndex.md` with 5 heuristic pages
- ✅ Updated `_ImplementationIndex.md` with Environment and Heuristic links
- ✅ Updated `_WorkflowIndex.md` with Environment and Heuristic links

## Code Evidence Sources

### Environment Constraints Found In:
| Source File | Constraint Type |
|-------------|-----------------|
| `unsloth/device_type.py` | GPU device detection (CUDA/HIP/XPU) |
| `unsloth/kernels/utils.py` | Triton version checks, bitsandbytes integration |
| `unsloth/save.py` | llama.cpp compilation, cmake/make detection |
| `unsloth/models/loader.py` | transformers version requirements |
| `pyproject.toml` | Package dependency versions |

### Heuristics Found In:
| Source File | Heuristic Type |
|-------------|----------------|
| `unsloth/trainer.py` | Sample packing, embedding LR, padding-free |
| `unsloth/save.py` | GGUF quantization methods, memory_usage |
| `unsloth/models/llama.py` | Dtype selection, bfloat16 checks |
| `unsloth/models/loader.py` | FORCE_FLOAT32 models list |
| `unsloth/kernels/utils.py` | Global buffer reuse |

## Statistics

- Environment pages created: 2
- Heuristic pages created: 5
- Total documentation lines: ~1,200
- Code evidence references: 25+
- Environment links added: 10
- Heuristic links added: 22

## Notes for Audit Phase

### Verified Connections
All Environment and Heuristic connections point to existing pages:
- All Implementation pages exist (7 total)
- All Workflow pages exist (3 total)
- All Principle pages exist (9 total)

### Potential Improvements
1. Could add more granular heuristics for specific model architectures
2. Vision model heuristics could be expanded
3. Multi-GPU training heuristics not fully documented

### No Broken Links Expected
- All `✅Type:Name` references verified against existing pages
- No `⬜Type:Name` (missing page) references in updated indexes
