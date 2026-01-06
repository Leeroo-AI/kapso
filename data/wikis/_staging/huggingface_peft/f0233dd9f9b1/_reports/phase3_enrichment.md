# Phase 3: Enrichment Report

## Summary

Phase 3 (Enrichment) extracted environment constraints and heuristics (tribal knowledge) from the huggingface_peft codebase. This phase creates leaf nodes in the knowledge graph that capture prerequisites and practical wisdom.

---

## Environments Created

| Environment | Required By | Description |
|-------------|-------------|-------------|
| huggingface_peft_Core_Environment | LoraConfig_init, get_peft_model, PeftModel_from_pretrained, PeftModel_save_pretrained, merge_and_unload, plus 10 more implementations | Base PEFT installation: Python>=3.10, torch>=1.13.0, transformers, accelerate>=0.21.0, huggingface_hub>=0.25.0 |
| huggingface_peft_Quantization_Environment | BitsAndBytesConfig_4bit, prepare_model_for_kbit_training | bitsandbytes for 4-bit/8-bit QLoRA training |
| huggingface_peft_GPTQ_Environment | get_peft_model | auto_gptq>=0.5.0 OR gptqmodel>=2.0.0+optimum>=1.24.0 for GPTQ quantization |
| huggingface_peft_LoftQ_Environment | LoraConfig_init | scipy for LoftQ initialization (SVD operations) |

### Key Findings - Environments

1. **Core Dependencies** (from `setup.py:60-71`):
   - torch >= 1.13.0
   - transformers (no version constraint)
   - accelerate >= 0.21.0
   - safetensors
   - huggingface_hub >= 0.25.0

2. **Optional Quantization Backends**:
   - bitsandbytes (4-bit/8-bit via `is_bnb_available()`, `is_bnb_4bit_available()`)
   - auto_gptq >= 0.5.0 OR gptqmodel >= 2.0.0 + optimum >= 1.24.0
   - scipy for LoftQ initialization

3. **Multi-platform Support**:
   - CUDA (primary)
   - MPS (Apple Silicon)
   - XPU (Intel)
   - NPU (Huawei Ascend)
   - MLU (Cambricon)
   - CPU fallback

---

## Heuristics Created

| Heuristic | Applies To | Description |
|-----------|------------|-------------|
| huggingface_peft_LoRA_Rank_Selection | LoraConfig_init, LoRA workflows | Guidelines for selecting `r` value: r=8 default, r=16-64 for complex tasks, use RSLoRA at r>=32 |
| huggingface_peft_Target_Module_Selection | LoraConfig_init, LoRA workflows | "all-linear" vs "q_proj,v_proj" module targeting strategies |
| huggingface_peft_Quantized_Merge_Warning | merge_and_unload, BitsAndBytesConfig_4bit | Warning: 4-bit/8-bit merges introduce rounding errors |
| huggingface_peft_Gradient_Checkpointing | prepare_model_for_kbit_training, Training workflows | Trade ~20% compute for 50-60% VRAM reduction |
| huggingface_peft_DoRA_Overhead | LoraConfig_init, LoRA workflows | DoRA improves quality at low ranks but adds overhead; merge for inference |

### Key Findings - Heuristics

1. **Tribal Knowledge from Code Comments**:
   - Tim Dettmers' note about defensive cloning for 4-bit (`bnb.py:548-553`)
   - RSLoRA scaling factor recommendation (`config.py:488-498`)
   - DoRA overhead warning (`config.py:634-645`)

2. **Common Warnings Extracted**:
   - "Merge lora module to 4-bit linear may get different generations due to rounding errors" (`bnb.py:397`)
   - "Merge lora module to 8-bit linear may get different generations due to rounding errors" (`bnb.py:110`)
   - "DoRA introduces a bigger overhead than pure LoRA" (`config.py:641-642`)

3. **Configuration Best Practices**:
   - Default `r=8` is suitable for most tasks
   - `target_modules="all-linear"` for maximum coverage
   - `use_rslora=True` stabilizes training at high ranks
   - `use_gradient_checkpointing=True` is default in `prepare_model_for_kbit_training()`

---

## Links Added

### Environment Links
- Updated _EnvironmentIndex.md with 4 new entries
- Updated _ImplementationIndex.md with Environment connections:
  - Core_Environment linked to 15 implementations
  - Quantization_Environment linked to 2 implementations
  - GPTQ_Environment linked to 1 implementation
  - LoftQ_Environment linked to 1 implementation

### Heuristic Links
- Updated _HeuristicIndex.md with 5 new entries
- Updated _ImplementationIndex.md with Heuristic connections:
  - LoRA_Rank_Selection linked to LoraConfig_init
  - Target_Module_Selection linked to LoraConfig_init
  - Quantized_Merge_Warning linked to merge_and_unload, BitsAndBytesConfig_4bit
  - Gradient_Checkpointing linked to prepare_model_for_kbit_training, Trainer_train
  - DoRA_Overhead linked to LoraConfig_init

---

## Notes for Audit Phase

### Pending Workflow Connections
The following workflow pages are referenced but not yet created:
- ⬜ huggingface_peft_LoRA_Fine_Tuning
- ⬜ huggingface_peft_QLoRA_Training

These should be created in a future workflow phase.

### Potential Improvements
1. Could add more environment variants (e.g., TPU-specific with torch_xla)
2. Could extract more heuristics from README/docs
3. GPTQ Environment could be split into AutoGPTQ vs GPTQModel variants

### Verification Status
- [x] All 4 Environment pages written to correct directory
- [x] All 5 Heuristic pages written to correct directory
- [x] _EnvironmentIndex.md updated
- [x] _HeuristicIndex.md updated
- [x] _ImplementationIndex.md updated with Env/Heuristic links
- [x] All filenames use `huggingface_peft_` prefix
- [x] No files created outside designated directories

---

## Statistics

| Category | Count |
|----------|-------|
| Environment Pages Created | 4 |
| Heuristic Pages Created | 5 |
| Implementation Pages Updated (in index) | 15 |
| Workflow References (pending) | 2 |

---

**Phase Complete:** 2024-12-18
