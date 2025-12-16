# Phase 1: Anchoring Report

## Summary
- **Workflows Created:** 3
- **Source Files Covered:** 37
- **Test Files Documented:** 15

## Workflows Created

| Workflow | Source Files | Steps |
|----------|--------------|-------|
| unslothai_unsloth_QLoRA_Finetuning | `loader.py`, `llama.py`, `trainer.py`, `save.py`, `chat_templates.py`, `packing.py`, `fast_lora.py`, Triton kernels | 6 steps |
| unslothai_unsloth_GGUF_Export | `save.py`, `tokenizer_utils.py`, `ollama_template_mappers.py`, `_utils.py`, `fast_lora.py` | 5 steps |
| unslothai_unsloth_GRPO_Training | `loader.py`, `rl.py`, `rl_replacements.py`, `chat_templates.py`, `save.py`, `fp8.py` | 7 steps |

## Coverage Summary

### Core Module Coverage
| Module | Files Covered | Workflows |
|--------|---------------|-----------|
| `unsloth/models/` | 8 files | QLoRA_Finetuning, GGUF_Export, GRPO_Training |
| `unsloth/kernels/` | 6 files | QLoRA_Finetuning, GRPO_Training |
| `unsloth/` (root) | 5 files | All workflows |
| `unsloth/utils/` | 3 files | QLoRA_Finetuning |

### Test File Coverage
| Test Category | Files | Workflow |
|---------------|-------|----------|
| QLoRA tests | 2 | QLoRA_Finetuning |
| Saving/merge tests | 10 | GGUF_Export |
| GRPO tests | 1 | GRPO_Training |
| Data/eval utils | 4 | Multiple |

## Principles to Create (Excavation Phase)

Based on the workflows, the following Principles need to be created:

### Shared Principles (used in multiple workflows)
1. **unslothai_unsloth_Model_Loading** - `FastLanguageModel.from_pretrained()` API
2. **unslothai_unsloth_LoRA_Configuration** - `get_peft_model()` and adapter setup
3. **unslothai_unsloth_Data_Formatting** - Chat templates and dataset preparation
4. **unslothai_unsloth_Model_Export** - `save_pretrained_merged()`, `push_to_hub_merged()`

### QLoRA-specific Principles
5. **unslothai_unsloth_Environment_Setup** - Installation and hardware detection
6. **unslothai_unsloth_SFT_Training** - SFTTrainer integration with Unsloth

### GGUF-specific Principles
7. **unslothai_unsloth_LoRA_Merging** - Adapter weight fusion
8. **unslothai_unsloth_GGUF_Conversion** - llama.cpp integration
9. **unslothai_unsloth_Ollama_Integration** - Modelfile generation
10. **unslothai_unsloth_Model_Deployment** - Hub and local deployment

### GRPO-specific Principles
11. **unslothai_unsloth_Reward_Functions** - Reward function design patterns
12. **unslothai_unsloth_GRPO_Training** - GRPOTrainer configuration

## Key APIs to Trace (for Excavation Phase)

### High Priority (Core User-Facing APIs)
1. `FastLanguageModel.from_pretrained()` → `unsloth/models/loader.py`
2. `FastLanguageModel.get_peft_model()` → `unsloth/models/loader.py`
3. `model.save_pretrained_merged()` → `unsloth/save.py`
4. `model.save_pretrained_gguf()` → `unsloth/save.py`
5. `model.push_to_hub_merged()` → `unsloth/save.py`

### Medium Priority (Internal Optimizations)
6. `get_chat_template()` → `unsloth/chat_templates.py`
7. `train_on_responses_only()` → `unsloth/chat_templates.py`
8. Triton kernel fusion → `unsloth/kernels/fast_lora.py`
9. Gradient checkpointing → `unsloth/models/llama.py`
10. Sequence packing → `unsloth/utils/packing.py`

### Lower Priority (RL-specific)
11. GRPOTrainer integration → `unsloth/models/rl.py`
12. vLLM fast inference → `unsloth/models/rl_replacements.py`
13. FP8 support → `unsloth/kernels/fp8.py`

## Important Classes/Functions

### Main Entry Points
- `FastLanguageModel` class (`loader.py:~50`)
- `FastModel` class (`loader.py`)
- `FastVisionModel` class (`loader.py`)

### Core Patching Functions
- `_prepare_model_for_kbit_training()` (`llama.py`)
- `patch_llama_model()` (`llama.py`)
- `get_peft_model()` (`loader.py`)

### Export Functions
- `unsloth_save_model()` (`save.py`)
- `save_to_gguf()` (`save.py`)
- `_merge_lora()` (`save.py:~182`)

### Optimization Functions
- `apply_lora_mlp()` (`fast_lora.py`)
- `apply_lora_qkv()` (`fast_lora.py`)
- `create_cross_entropy_loss()` (`cross_entropy_loss.py`)

## Recommendations for Next Phase

1. **Start with Model_Loading Principle** - Most fundamental, used by all workflows
2. **Create Implementation pages for core classes** - `FastLanguageModel`, `FastModel`
3. **Document the monkey-patching strategy** - Key architectural pattern
4. **Trace quantization flow** - BitsAndBytes 4-bit → Unsloth optimizations → GGUF export
5. **Map TRL integration points** - How Unsloth extends SFTTrainer/GRPOTrainer

## Files Not Yet Covered

The following files are not directly covered by current workflows but may warrant future documentation:

- Vision model support (`vision.py`, vision tests)
- Text-to-speech models (`test_csm.py`, `test_orpheus.py`, `test_whisper.py`)
- Synthetic data generation (`dataprep/synthetic.py`)
- MoE (Mixture of Experts) kernels (`kernels/moe/`)
- Model registry system (`registry/`)
