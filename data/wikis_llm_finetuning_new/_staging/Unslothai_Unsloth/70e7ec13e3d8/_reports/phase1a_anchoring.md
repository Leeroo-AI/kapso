# Phase 1a: Anchoring Report

## Summary

- **Workflows created:** 4
- **Total steps documented:** 25

## Workflows Created

| Workflow | Source Files | Steps | Rough APIs |
|----------|--------------|-------|------------|
| QLoRA_Finetuning | loader.py, llama.py, chat_templates.py, trainer.py, save.py | 6 | FastLanguageModel.from_pretrained, get_peft_model, get_chat_template, SFTTrainer, save_pretrained_merged |
| GRPO_Reinforcement_Learning | loader.py, llama.py, rl.py, rl_replacements.py, chat_templates.py, save.py | 7 | FastLanguageModel.from_pretrained, get_peft_model, SFTTrainer, GRPOTrainer, save_pretrained_merged |
| Vision_Model_Finetuning | loader.py, vision.py, trainer.py, save.py | 6 | FastVisionModel.from_pretrained, get_peft_model, UnslothVisionDataCollator, SFTTrainer, save_pretrained_merged |
| GGUF_Export | save.py, ollama_template_mappers.py, tokenizer_utils.py | 6 | save_pretrained_merged, convert_to_gguf, quantize_gguf, OLLAMA_TEMPLATES |

## Coverage Summary

- **Source files covered:** 35 (core package files with workflow coverage)
- **Example/test files documented:** 10 (QLoRA tests, GRPO test, Vision OCR benchmarks)

## Source Files Identified Per Workflow

### Unslothai_Unsloth_QLoRA_Finetuning

| File | Purpose |
|------|---------|
| `unsloth/models/loader.py` | FastLanguageModel.from_pretrained entry point |
| `unsloth/models/llama.py` | FastLlamaModel.get_peft_model, LoRA injection |
| `unsloth/chat_templates.py` | get_chat_template, train_on_responses_only |
| `unsloth/trainer.py` | UnslothTrainer, SFTTrainer integration |
| `unsloth/save.py` | unsloth_save_model, _merge_lora, save_pretrained_merged |
| `unsloth/models/_utils.py` | Core utilities for model patching |
| `unsloth/kernels/fast_lora.py` | Fused LoRA operations (Triton) |
| `unsloth/kernels/cross_entropy_loss.py` | Optimized cross-entropy kernel |
| `tests/qlora/test_unsloth_qlora_train_and_merge.py` | Complete example workflow |

### Unslothai_Unsloth_GRPO_Reinforcement_Learning

| File | Purpose |
|------|---------|
| `unsloth/models/loader.py` | FastLanguageModel with fast_inference mode |
| `unsloth/models/rl.py` | PatchFastRL, TRL trainer patching |
| `unsloth/models/rl_replacements.py` | RL_FUNCTIONS, RL_CONFIG_CHANGES |
| `unsloth/chat_templates.py` | get_chat_template, train_on_responses_only |
| `unsloth/trainer.py` | SFTTrainer for warmup stage |
| `unsloth/save.py` | Model merging and saving |
| `tests/saving/language_models/test_save_merged_grpo_model.py` | Complete GRPO example |
| `tests/utils/aime_eval.py` | AIME benchmark evaluation |

### Unslothai_Unsloth_Vision_Model_Finetuning

| File | Purpose |
|------|---------|
| `unsloth/models/vision.py` | FastVisionModel, FastBaseModel |
| `unsloth/models/loader.py` | FastVisionModel.from_pretrained dispatcher |
| `unsloth/trainer.py` | UnslothVisionDataCollator |
| `unsloth/save.py` | Vision model saving |
| `tests/saving/vision_models/test_save_merge_vision_model_ocr_benchmark.py` | Complete vision example |
| `tests/utils/ocr_eval.py` | OCR evaluation utilities (WER/CER) |

### Unslothai_Unsloth_GGUF_Export

| File | Purpose |
|------|---------|
| `unsloth/save.py` | save_to_gguf, ALLOWED_QUANTS, GGUF conversion |
| `unsloth/ollama_template_mappers.py` | OLLAMA_TEMPLATES, MODEL_TO_OLLAMA_TEMPLATE_MAPPER |
| `unsloth/tokenizer_utils.py` | fix_sentencepiece_gguf |
| `tests/saving/test_unsloth_save.py` | Save functionality tests |

## Notes for Phase 1b (Enrichment)

### Files Requiring Line-by-Line Tracing

1. **`unsloth/models/loader.py`** - Critical for Model_Loading step
   - Trace `FastLanguageModel.from_pretrained` (L121-675)
   - Document model architecture detection logic
   - Map quantization configuration flow

2. **`unsloth/models/llama.py`** - Critical for LoRA_Adapter_Injection step
   - Trace `FastLlamaModel.get_peft_model`
   - Document target_modules selection
   - Map gradient checkpointing configuration

3. **`unsloth/save.py`** - Critical for Model_Saving and GGUF_Export steps
   - Trace `unsloth_save_model` (L234-300)
   - Document `_merge_lora` function
   - Map GGUF conversion flow

4. **`unsloth/models/rl.py`** - Critical for GRPO_Training step
   - Trace `PatchFastRL` function
   - Document `unsloth_unwrap_model_for_generation`
   - Map RL trainer patching

### External APIs to Document

| API | Library | Usage in Workflows |
|-----|---------|-------------------|
| `SFTTrainer` | trl | QLoRA_Finetuning, Vision_Model_Finetuning, GRPO_Reinforcement_Learning |
| `GRPOTrainer` | trl | GRPO_Reinforcement_Learning |
| `GRPOConfig` | trl | GRPO_Reinforcement_Learning |
| `get_peft_model` | peft | All fine-tuning workflows |
| `AutoModelForCausalLM` | transformers | Model loading (wrapped by Unsloth) |
| `BitsAndBytesConfig` | transformers/bitsandbytes | 4-bit quantization |
| `convert_to_gguf` | unsloth_zoo | GGUF_Export |
| `quantize_gguf` | unsloth_zoo | GGUF_Export |

### Unclear Mappings / Questions for Enrichment

1. **Vision model architecture detection** - How does FastVisionModel determine the correct model class (Qwen2VL, Llama3.2Vision, Pixtral)?

2. **GGUF conversion dependencies** - The actual GGUF conversion logic is in `unsloth_zoo`, need to trace the integration points.

3. **Reward function interface for GRPO** - Document the expected signature and return values for custom reward functions.

4. **Embedding learning rate configuration** - The `embedding_learning_rate` parameter in UnslothTrainingArguments needs documentation.

## Workflow Files Created

```
/home/ubuntu/praxium/data/wikis_llm_finetuning_new/_staging/Unslothai_Unsloth/70e7ec13e3d8/workflows/
├── Unslothai_Unsloth_QLoRA_Finetuning.md
├── Unslothai_Unsloth_GRPO_Reinforcement_Learning.md
├── Unslothai_Unsloth_Vision_Model_Finetuning.md
└── Unslothai_Unsloth_GGUF_Export.md
```

## Index Files Updated

- `_RepoMap_Unslothai_Unsloth.md` - Coverage column populated for all relevant files
- `_WorkflowIndex.md` - All 4 workflows documented with rough structure

## Phase 1a Completion Status

- [x] Read Phase 0 report
- [x] Read Repository Map index
- [x] Scan high-level documentation
- [x] Identify candidate workflows (4 identified)
- [x] Write Workflow pages (4 created)
- [x] Update Coverage in Repository Map
- [x] Write rough WorkflowIndex
- [x] Write execution report
