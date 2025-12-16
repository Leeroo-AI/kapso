# Phase 1: Anchoring Report

## Summary

Phase 1 identified and documented 4 key workflows representing the "Golden Paths" for the Unsloth repository. These workflows cover the primary use cases: QLoRA fine-tuning, vision-language model training, GGUF export, and GRPO reinforcement learning.

## Workflows Created

| Workflow | Source Files | Steps |
|----------|--------------|-------|
| unslothai_unsloth_QLoRA_Finetuning | `unsloth/models/loader.py`, `unsloth/save.py`, `unsloth/trainer.py`, `tests/qlora/` | 6 |
| unslothai_unsloth_Vision_Language_Model_Finetuning | `unsloth/models/vision.py`, `unsloth/trainer.py`, `tests/saving/vision_models/` | 6 |
| unslothai_unsloth_GGUF_Export | `unsloth/save.py`, `unsloth/tokenizer_utils.py`, `unsloth/ollama_template_mappers.py` | 6 |
| unslothai_unsloth_GRPO_Reinforcement_Learning | `unsloth/models/rl.py`, `unsloth/models/rl_replacements.py`, `tests/saving/language_models/test_save_merged_grpo_model.py` | 8 |

## Coverage Summary

- **Source files covered:** 32 files now have Workflow coverage annotations
- **Example files documented:** No traditional example scripts exist; coverage is based on test files
- **Key packages covered:**
  - `unsloth/models/` - Model loading, optimization, RL patching
  - `unsloth/kernels/` - Optimized operations (cross-entropy, LoRA, RoPE)
  - `unsloth/` - Save, tokenizer, trainer modules

## Workflow Details

### 1. QLoRA Fine-tuning Workflow
**Primary use case** for Unsloth - loading models in 4-bit quantization and training with LoRA adapters.

**Key Steps:**
1. Package Initialization
2. Model Loading (4-bit quantization)
3. LoRA Injection
4. Data Formatting (chat templates)
5. SFT Training
6. Model Saving

**Key APIs to trace:**
- `FastLanguageModel.from_pretrained()`
- `FastLanguageModel.get_peft_model()`
- `model.save_pretrained()` / `model.save_pretrained_merged()`

### 2. Vision-Language Model Fine-tuning Workflow
Training multimodal models like Qwen2.5-VL and LLaVA.

**Key Steps:**
1. Package Initialization
2. Vision Model Loading
3. Vision LoRA Injection
4. Vision Data Formatting
5. Vision SFT Training
6. Vision Model Saving

**Key APIs to trace:**
- `FastVisionModel.from_pretrained()`
- `FastVisionModel.get_peft_model()`
- `UnslothVisionDataCollator`

### 3. GGUF Export Workflow
Exporting trained models to GGUF format for llama.cpp/Ollama deployment.

**Key Steps:**
1. Model Preparation
2. Weight Merging
3. GGUF Conversion
4. GGUF Validation
5. Hub Upload
6. Ollama Integration

**Key APIs to trace:**
- `model.save_pretrained_merged()`
- `model.save_pretrained_gguf()`
- `model.push_to_hub_gguf()`
- `convert_to_gguf()`, `quantize_gguf()` from `unsloth_zoo.llama_cpp`

### 4. GRPO Reinforcement Learning Workflow
Reinforcement learning training using GRPO and related algorithms.

**Key Steps:**
1. RL Setup (PatchFastRL)
2. RL Model Loading (with vLLM)
3. RL LoRA Setup
4. RL Data Preparation
5. Reward Definition
6. GRPO Configuration
7. GRPO Training
8. RL Model Saving

**Key APIs to trace:**
- `PatchFastRL()`
- `GRPOTrainer`, `GRPOConfig` from TRL
- vLLM integration via `fast_inference=True`

## Notes for Excavation Phase

### Key APIs to Trace from Workflows

1. **FastLanguageModel** (high priority)
   - `from_pretrained()` - in `unsloth/models/loader.py`
   - `get_peft_model()` - in `unsloth/models/loader.py`
   - `for_inference()` / `for_training()` - training mode switching

2. **FastVisionModel** (high priority)
   - Similar API surface to FastLanguageModel
   - Located in `unsloth/models/vision.py`

3. **PatchFastRL** (high priority)
   - Entry point for RL training
   - Located in `unsloth/models/rl.py`
   - Patches TRL trainers dynamically

4. **Save Functions** (high priority)
   - `unsloth_save_model()` - core saving logic
   - `save_to_gguf()` - GGUF conversion
   - Located in `unsloth/save.py`

### Important Classes/Functions Used

| Function/Class | Location | Used By Workflow |
|----------------|----------|------------------|
| `FastLanguageModel` | `unsloth/__init__.py` | QLoRA, GRPO |
| `FastVisionModel` | `unsloth/__init__.py` | Vision |
| `PatchFastRL` | `unsloth/models/rl.py` | GRPO |
| `unsloth_save_model` | `unsloth/save.py` | All |
| `save_to_gguf` | `unsloth/save.py` | GGUF Export |
| `get_chat_template` | `unsloth/chat_templates.py` | QLoRA |
| `UnslothVisionDataCollator` | `unsloth/trainer.py` | Vision |

### Kernel Implementations to Document

- `unsloth/kernels/cross_entropy_loss.py` - Chunked cross-entropy
- `unsloth/kernels/fast_lora.py` - Fused LoRA operations
- `unsloth/kernels/rms_layernorm.py` - Fused RMS normalization
- `unsloth/kernels/rope_embedding.py` - Optimized RoPE

### Model Architecture Mappings

The `unsloth/models/` directory contains architecture-specific optimizations:
- `llama.py` - Core Llama/Llama2/Llama3 support
- `mistral.py` - Mistral with sliding window attention
- `gemma.py`, `gemma2.py` - Google Gemma models
- `qwen2.py`, `qwen3.py` - Qwen family
- `cohere.py` - Cohere Command models

## Principle Pages to Create (Next Phase)

Based on the workflow steps, the following Principle pages should be created:

1. `unslothai_unsloth_Package_Initialization`
2. `unslothai_unsloth_Model_Loading`
3. `unslothai_unsloth_LoRA_Injection`
4. `unslothai_unsloth_Data_Formatting`
5. `unslothai_unsloth_SFT_Training`
6. `unslothai_unsloth_Model_Saving`
7. `unslothai_unsloth_Vision_Model_Loading`
8. `unslothai_unsloth_Vision_LoRA_Injection`
9. `unslothai_unsloth_Vision_Data_Formatting`
10. `unslothai_unsloth_Vision_SFT_Training`
11. `unslothai_unsloth_Vision_Model_Saving`
12. `unslothai_unsloth_Model_Preparation`
13. `unslothai_unsloth_Weight_Merging`
14. `unslothai_unsloth_GGUF_Conversion`
15. `unslothai_unsloth_GGUF_Validation`
16. `unslothai_unsloth_Hub_Upload`
17. `unslothai_unsloth_Ollama_Integration`
18. `unslothai_unsloth_RL_Setup`
19. `unslothai_unsloth_RL_Model_Loading`
20. `unslothai_unsloth_RL_LoRA_Setup`
21. `unslothai_unsloth_RL_Data_Preparation`
22. `unslothai_unsloth_Reward_Definition`
23. `unslothai_unsloth_GRPO_Configuration`
24. `unslothai_unsloth_GRPO_Training`
25. `unslothai_unsloth_RL_Model_Saving`

---

*Generated: 2025-12-16*
