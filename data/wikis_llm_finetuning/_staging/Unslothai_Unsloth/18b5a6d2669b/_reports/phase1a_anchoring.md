# Phase 1a: Anchoring Report

## Summary
- Workflows created: 4
- Total steps documented: 28

## Workflows Created

| Workflow | Source Files | Steps | Rough APIs |
|----------|--------------|-------|------------|
| QLoRA_Finetuning | loader.py, llama.py, chat_templates.py, trainer.py, save.py | 6 | `FastLanguageModel.from_pretrained`, `get_peft_model`, `get_chat_template`, `SFTTrainer`, `save_pretrained_merged` |
| GRPO_Training | loader.py, rl.py, rl_replacements.py, chat_templates.py, save.py | 9 | `FastLanguageModel.from_pretrained`, `GRPOTrainer`, `train_on_responses_only` |
| Vision_Finetuning | vision.py, trainer.py, save.py | 7 | `FastVisionModel.from_pretrained`, `get_peft_model`, `UnslothVisionDataCollator` |
| GGUF_Export | save.py, ollama_template_mappers.py, tokenizer_utils.py | 6 | `save_pretrained_gguf`, `push_to_hub_gguf`, `convert_to_gguf` |

## Coverage Summary
- Source files covered: 42 (with workflow attribution in Coverage column)
- Key package files: loader.py, llama.py, save.py, chat_templates.py, vision.py, rl.py
- Test files documented: 15 (showing workflow relationships)

## Source Files Identified Per Workflow

### Unslothai_Unsloth_QLoRA_Finetuning
- `unsloth/models/loader.py` - FastLanguageModel.from_pretrained implementation (L123-250)
- `unsloth/models/llama.py` - get_peft_model and base model patching (3,452 lines)
- `unsloth/chat_templates.py` - Chat template definitions (3,159 lines)
- `unsloth/trainer.py` - UnslothTrainer class (438 lines)
- `unsloth/save.py` - Model saving functions (3,100 lines)
- `unsloth/kernels/fast_lora.py` - LoRA autograd functions (730 lines)
- `tests/qlora/test_unsloth_qlora_train_and_merge.py` - Reference example

### Unslothai_Unsloth_GRPO_Training
- `unsloth/models/loader.py` - fast_inference parameter for vLLM integration
- `unsloth/models/rl.py` - RL trainer patches (1,443 lines)
- `unsloth/models/rl_replacements.py` - RL method patches (995 lines)
- `unsloth/chat_templates.py` - train_on_responses_only function
- `tests/saving/language_models/test_save_merged_grpo_model.py` - Full GRPO example (825 lines)
- `tests/utils/aime_eval.py` - AIME evaluation utilities

### Unslothai_Unsloth_Vision_Finetuning
- `unsloth/models/vision.py` - FastVisionModel implementation (1,292 lines)
- `unsloth/trainer.py` - UnslothVisionDataCollator
- `tests/saving/vision_models/test_save_merge_vision_model_ocr_benchmark.py` - VLM example
- `tests/utils/ocr_eval.py` - OCR evaluation metrics

### Unslothai_Unsloth_GGUF_Export
- `unsloth/save.py` - save_pretrained_gguf, save_to_gguf, push_to_hub_gguf
- `unsloth/ollama_template_mappers.py` - OLLAMA_TEMPLATES, MODEL_TO_OLLAMA_TEMPLATE_MAPPER (2,192 lines)
- `unsloth/tokenizer_utils.py` - fix_sentencepiece_gguf (1,106 lines)

## Principles Identified (for Phase 2)

### QLoRA_Finetuning Workflow
1. **Model_Loading** - `FastLanguageModel.from_pretrained` API
2. **LoRA_Configuration** - `FastLanguageModel.get_peft_model` API
3. **Data_Formatting** - `get_chat_template`, `apply_chat_template`
4. **Training_Configuration** - SFTConfig wrapper (TRL)
5. **Supervised_Finetuning** - SFTTrainer execution
6. **Model_Saving** - `save_pretrained_merged`, `save_pretrained`

### GRPO_Training Workflow
1. **RL_Model_Loading** - `from_pretrained(fast_inference=True)` for vLLM
2. **LoRA_Configuration** - Shared with QLoRA
3. **Chat_Template_Configuration** - Reasoning format setup
4. **RL_Dataset_Preparation** - User-defined dataset mapping
5. **Reward_Functions** - User-defined reward interface
6. **SFT_Pretraining** - Optional warm-start with `train_on_responses_only`
7. **GRPO_Configuration** - GRPOConfig from TRL
8. **GRPO_Execution** - GRPOTrainer.train integration
9. **Model_Saving** - Shared with QLoRA

### Vision_Finetuning Workflow
1. **Vision_Model_Loading** - `FastVisionModel.from_pretrained`
2. **Vision_LoRA_Configuration** - Vision-specific LoRA params (finetune_vision_layers, etc.)
3. **Multimodal_Data_Preparation** - Message format with images
4. **Vision_Training_Mode** - `FastVisionModel.for_training`
5. **Vision_Training** - `UnslothVisionDataCollator` integration
6. **Vision_Inference_Mode** - `FastVisionModel.for_inference`
7. **Vision_Model_Saving** - Same save APIs as text models

### GGUF_Export Workflow
1. **Model_Preparation** - Pre-export model state
2. **Quantization_Selection** - ALLOWED_QUANTS mapping
3. **GGUF_Export** - `save_pretrained_gguf` execution
4. **Ollama_Modelfile** - Template mapping to Ollama format
5. **GGUF_Hub_Upload** - `push_to_hub_gguf`
6. **GGUF_Verification** - External llama-cli testing

## Notes for Phase 1b (Enrichment)

### Files Needing Line-by-Line Tracing
1. `unsloth/models/loader.py:L123-600` - from_pretrained full flow
2. `unsloth/models/llama.py:L2000-3000` - get_peft_model implementation
3. `unsloth/save.py:L1785-2000` - save_pretrained_gguf flow
4. `unsloth/models/rl.py:L1-500` - GRPO trainer patches

### External APIs to Document
- **TRL Library**: `SFTTrainer`, `SFTConfig`, `GRPOTrainer`, `GRPOConfig`
- **PEFT Library**: `get_peft_model` (wrapped by Unsloth)
- **bitsandbytes**: 4-bit/8-bit quantization
- **vLLM**: Fast inference backend (when `fast_inference=True`)
- **llama.cpp**: GGUF conversion and quantization

### Unclear Mappings
1. The relationship between `FastLanguageModel` and `FastModel` needs clarification - they share `from_pretrained` but route to different implementations based on parameters
2. Vision model saving may have different code paths in `save.py` (look for VLM-specific handling)
3. GRPO reward function interface is user-defined but follows specific signature patterns

### Shared Principles
Several principles are shared across workflows:
- **LoRA_Configuration**: QLoRA, GRPO, Vision all use similar `get_peft_model` patterns
- **Model_Saving**: All workflows end with same save APIs
- **Chat_Template_Configuration**: Shared between QLoRA and GRPO

Phase 2 should create these as reusable Principle pages linked from multiple workflows.

---

*Report generated: 2026-01-09*
*Phase 1a Status: COMPLETE*
