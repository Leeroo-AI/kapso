# Phase 3: Enrichment Report

## Summary

This phase mined the Unsloth codebase for **Environment constraints** (hardware, dependencies, credentials) and **Heuristics** (tribal knowledge, tips, best practices). Created 3 Environment pages and 6 Heuristic pages, then linked them to relevant Implementation pages.

## Environments Created

| Environment | Required By | Notes |
|-------------|-------------|-------|
| Unslothai_Unsloth_CUDA_GPU_Environment | 18 implementations | Base GPU environment for NVIDIA CUDA, AMD ROCm, Intel XPU |
| Unslothai_Unsloth_CUDA_GPU_vLLM_Environment | 7 implementations | Extended env with vLLM for GRPO/RL training |
| Unslothai_Unsloth_llama_cpp_Environment | 1 implementation | llama.cpp for GGUF export/validation |

### Environment Details

#### CUDA_GPU_Environment
- **Key Dependencies:** torch>=2.4.0, transformers>=4.45.0, bitsandbytes>=0.43.3, triton>=3.0.0, peft>=0.10.0
- **Hardware:** NVIDIA GPU (CUDA 11.8+), AMD GPU (ROCm 6.0+), or Intel XPU (PyTorch 2.6+)
- **Code Evidence:** `device_type.py:37-59` (device detection), `kernels/utils.py:41-44` (XPU version check)

#### CUDA_GPU_vLLM_Environment
- **Key Dependencies:** All from base + vllm>=0.6.0
- **Hardware:** NVIDIA GPU (24GB+ VRAM recommended)
- **Code Evidence:** `loader.py:234-249` (vLLM availability check, DGX Spark compatibility)

#### llama_cpp_Environment
- **Key Dependencies:** llama.cpp (built), sentencepiece, psutil
- **Hardware:** CPU sufficient, GPU optional for faster quantization
- **Code Evidence:** `save.py:104-131` (ALLOWED_QUANTS dictionary)

## Heuristics Created

| Heuristic | Applies To | Summary |
|-----------|------------|---------|
| Unslothai_Unsloth_Gradient_Checkpointing_Tip | FastLanguageModel, FastVisionModel | Use `"unsloth"` mode for 50-60% VRAM reduction |
| Unslothai_Unsloth_LoRA_Rank_Selection_Tip | get_peft_model, UnslothGRPOTrainer | r=16 default; r<=64 for vLLM compatibility |
| Unslothai_Unsloth_Sample_Packing_Tip | SFTTrainer, UnslothTrainingArguments | `packing=True` for >2x training speedup |
| Unslothai_Unsloth_Embedding_Learning_Rate_Tip | UnslothTrainingArguments | Use 5e-5 for embeddings vs 2e-4 for LoRA |
| Unslothai_Unsloth_GGUF_Quantization_Selection_Tip | save_to_gguf, push_to_hub_gguf | q4_k_m balanced; q8_0 for quality; q2_k for compression |
| Unslothai_Unsloth_BFloat16_vs_Float16_Tip | FastLanguageModel, FastVisionModel | BFloat16 on Ampere+ GPUs; Float16 on older GPUs |

### Heuristic Details

#### Gradient_Checkpointing_Tip
- **Source:** `loader.py:562-563`, default parameter in `from_pretrained`
- **Insight:** Unsloth's custom implementation trades ~20% speed for ~50-60% VRAM reduction

#### LoRA_Rank_Selection_Tip
- **Source:** `loader.py:795-803` (default modules), `rl.py:145-155` (vLLM rank constraint)
- **Insight:** r=16 works for most tasks; higher ranks for complex tasks but <= 64 for vLLM

#### Sample_Packing_Tip
- **Source:** `trainer.py:57-60` (blocklist), `trainer.py:393-396` (user message)
- **Insight:** Packing concatenates short samples for efficient GPU utilization

#### Embedding_Learning_Rate_Tip
- **Source:** `trainer.py:139-179` (`_create_unsloth_optimizer`)
- **Insight:** Embeddings need lower LR (5e-5) than LoRA parameters (2e-4)

#### GGUF_Quantization_Selection_Tip
- **Source:** `save.py:104-131` (ALLOWED_QUANTS)
- **Insight:** q4_k_m is balanced default; q8_0 near-lossless; q2_k max compression

#### BFloat16_vs_Float16_Tip
- **Source:** `_utils.py:154-165` (`is_bfloat16_supported`), auto-detection in loader
- **Insight:** BFloat16 prevents overflow, more stable; auto-detected by Unsloth

## Links Added

### Environment Links
- **CUDA_GPU_Environment:** Added to 18 implementations (all QLoRA, Vision, and GGUF pages)
- **CUDA_GPU_vLLM_Environment:** Already present in 7 GRPO implementations
- **llama_cpp_Environment:** Already present in llama_cli_validation

### Heuristic Links
- **Gradient_Checkpointing_Tip:** Added to 3 implementations
- **LoRA_Rank_Selection_Tip:** Added to 4 implementations
- **Sample_Packing_Tip:** Added to 2 implementations
- **Embedding_Learning_Rate_Tip:** Added to 1 implementation
- **GGUF_Quantization_Selection_Tip:** Added to 2 implementations (save_to_gguf, push_to_hub_gguf)
- **BFloat16_vs_Float16_Tip:** Added to 2 implementations

## Index Updates

### _EnvironmentIndex.md
- Added 3 Environment entries with full connection tracking

### _HeuristicIndex.md
- Added 6 Heuristic entries with full connection tracking

### _ImplementationIndex.md
- Environment references were already `✅Env:` (correctly linked)
- Heuristic links were added/updated in individual implementation files

## Code Patterns Scanned

### Environment Constraints Found
1. **Device detection:** `torch.cuda.is_available()`, `torch.xpu.is_available()`, `is_hip()` in device_type.py
2. **Version checks:** `Version(torch.__version__) < Version("2.6.0")` for Intel XPU
3. **Dependency checks:** `importlib.util.find_spec("vllm")` for vLLM availability
4. **Hardware compatibility:** DGX Spark (GB10) check in loader.py

### Heuristics Found
1. **Comments with tips:** `trainer.py` packing blocklist with reasons
2. **Default values:** `use_gradient_checkpointing="unsloth"`, `r=16`, `lora_alpha=16`
3. **Warning messages:** "Packing enabled - training is >2x faster"
4. **Config dictionaries:** `ALLOWED_QUANTS` with descriptions

## Notes for Audit Phase

### Potential Issues
1. **Heuristic links with wrong names:** Several implementation files had heuristic links to non-existent pages (e.g., `Unslothai_Unsloth_Memory_Optimization`, `Unslothai_Unsloth_vLLM_Memory_Tuning`, `Unslothai_Unsloth_GRPO_Hyperparameters`). These were updated to point to actual created heuristic pages.

2. **Missing heuristics that could be created:**
   - GRPO hyperparameters (beta, num_generations, etc.) - could be extracted from rl.py
   - vLLM memory tuning (`gpu_memory_utilization` parameter) - partial coverage in existing pages

### Verification Needed
- Confirm all `[[requires_env::...]]` links in implementation files point to valid environment pages
- Confirm all `[[uses_heuristic::...]]` links point to valid heuristic pages
- Verify backlinks in environment/heuristic pages match forward links in implementation pages

## Files Created

### Environments (3 files)
- `/environments/Unslothai_Unsloth_CUDA_GPU_Environment.md`
- `/environments/Unslothai_Unsloth_CUDA_GPU_vLLM_Environment.md`
- `/environments/Unslothai_Unsloth_llama_cpp_Environment.md`

### Heuristics (6 files)
- `/heuristics/Unslothai_Unsloth_Gradient_Checkpointing_Tip.md`
- `/heuristics/Unslothai_Unsloth_LoRA_Rank_Selection_Tip.md`
- `/heuristics/Unslothai_Unsloth_Sample_Packing_Tip.md`
- `/heuristics/Unslothai_Unsloth_Embedding_Learning_Rate_Tip.md`
- `/heuristics/Unslothai_Unsloth_GGUF_Quantization_Selection_Tip.md`
- `/heuristics/Unslothai_Unsloth_BFloat16_vs_Float16_Tip.md`

### Indexes Updated (2 files)
- `/_EnvironmentIndex.md`
- `/_HeuristicIndex.md`

### Implementation Files Updated (11 files)
- `Unslothai_Unsloth_FastLanguageModel_from_pretrained.md`
- `Unslothai_Unsloth_get_peft_model.md`
- `Unslothai_Unsloth_SFTTrainer_train.md`
- `Unslothai_Unsloth_UnslothTrainingArguments.md`
- `Unslothai_Unsloth_save_to_gguf.md`
- `Unslothai_Unsloth_FastLanguageModel_from_pretrained_vllm.md`
- `Unslothai_Unsloth_UnslothGRPOTrainer.md`
- `Unslothai_Unsloth_FastVisionModel_from_pretrained.md`
- `Unslothai_Unsloth_push_to_hub_gguf.md`
