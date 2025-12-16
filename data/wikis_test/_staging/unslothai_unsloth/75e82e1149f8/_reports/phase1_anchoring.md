# Phase 1: Anchoring Report

## Summary
- **Workflows Created:** 3
- **Source Files Covered:** 35
- **Test Files Documented:** 10

## Workflows Created

| Workflow | Source Files | Steps |
|----------|--------------|-------|
| unslothai_unsloth_QLoRA_Finetuning | `unsloth/models/loader.py`, `unsloth/models/llama.py`, `unsloth/chat_templates.py`, `unsloth/trainer.py`, `unsloth/save.py`, `tests/qlora/test_unsloth_qlora_train_and_merge.py` | 7 |
| unslothai_unsloth_Model_Export_GGUF | `unsloth/save.py`, `unsloth/ollama_template_mappers.py`, `unsloth/tokenizer_utils.py`, `tests/saving/test_unsloth_save.py` | 6 |
| unslothai_unsloth_Vision_Model_Finetuning | `unsloth/models/vision.py`, `unsloth/models/loader.py`, `unsloth/trainer.py`, `tests/saving/vision_models/test_push_to_hub_merged.py` | 7 |

## Coverage Summary

### Package Files Covered
| File | Workflows |
|------|-----------|
| `unsloth/__init__.py` | QLoRA_Finetuning |
| `unsloth/chat_templates.py` | QLoRA_Finetuning |
| `unsloth/kernels/__init__.py` | QLoRA_Finetuning |
| `unsloth/kernels/cross_entropy_loss.py` | QLoRA_Finetuning |
| `unsloth/kernels/fast_lora.py` | QLoRA_Finetuning |
| `unsloth/kernels/rms_layernorm.py` | QLoRA_Finetuning |
| `unsloth/kernels/rope_embedding.py` | QLoRA_Finetuning |
| `unsloth/kernels/swiglu.py` | QLoRA_Finetuning |
| `unsloth/kernels/utils.py` | QLoRA_Finetuning |
| `unsloth/models/_utils.py` | QLoRA_Finetuning |
| `unsloth/models/llama.py` | QLoRA_Finetuning |
| `unsloth/models/loader.py` | QLoRA_Finetuning, Model_Export_GGUF, Vision_Model_Finetuning |
| `unsloth/models/loader_utils.py` | QLoRA_Finetuning |
| `unsloth/models/mapper.py` | QLoRA_Finetuning |
| `unsloth/models/rl.py` | QLoRA_Finetuning |
| `unsloth/models/vision.py` | Vision_Model_Finetuning |
| `unsloth/ollama_template_mappers.py` | Model_Export_GGUF |
| `unsloth/save.py` | QLoRA_Finetuning, Model_Export_GGUF |
| `unsloth/tokenizer_utils.py` | QLoRA_Finetuning, Model_Export_GGUF |
| `unsloth/trainer.py` | QLoRA_Finetuning, Vision_Model_Finetuning |
| `unsloth/utils/hf_hub.py` | Model_Export_GGUF |
| `unsloth/utils/packing.py` | QLoRA_Finetuning |

### Test Files Covered
| File | Workflows |
|------|-----------|
| `tests/qlora/test_hf_qlora_train_and_merge.py` | QLoRA_Finetuning |
| `tests/qlora/test_unsloth_qlora_train_and_merge.py` | QLoRA_Finetuning |
| `tests/saving/language_models/test_merge_4bit_validation.py` | Model_Export_GGUF |
| `tests/saving/language_models/test_push_to_hub_merged.py` | Model_Export_GGUF |
| `tests/saving/language_models/test_save_merged_grpo_model.py` | QLoRA_Finetuning |
| `tests/saving/test_unsloth_save.py` | Model_Export_GGUF |
| `tests/saving/vision_models/test_push_to_hub_merged.py` | Vision_Model_Finetuning |
| `tests/saving/vision_models/test_save_merge_qwen2.5vl32B_model_ocr_benchmark.py` | Vision_Model_Finetuning |
| `tests/saving/vision_models/test_save_merge_vision_model_ocr_benchmark.py` | Vision_Model_Finetuning |
| `tests/utils/data_utils.py` | QLoRA_Finetuning |
| `tests/utils/ocr_eval.py` | Vision_Model_Finetuning |
| `tests/utils/test_packing.py` | QLoRA_Finetuning |

## Notes for Excavation Phase

### Key APIs to Trace from Workflows

#### From QLoRA_Finetuning:
- `FastLanguageModel.from_pretrained()` - Entry point for model loading in `unsloth/models/loader.py:122`
- `FastLanguageModel.get_peft_model()` - LoRA adapter injection in `unsloth/models/llama.py`
- `get_chat_template()` - Chat template application in `unsloth/chat_templates.py`
- `train_on_responses_only()` - Response masking in `unsloth/chat_templates.py`
- `model.save_pretrained_merged()` - Model saving in `unsloth/save.py`

#### From Model_Export_GGUF:
- `unsloth_save_model()` - Core save function in `unsloth/save.py:228`
- `save_to_gguf()` - GGUF conversion in `unsloth/save.py`
- `_merge_lora()` - LoRA weight merging in `unsloth/save.py:182`
- `OLLAMA_TEMPLATES` - Chat template mappings in `unsloth/ollama_template_mappers.py`
- `fix_sentencepiece_gguf()` - Tokenizer fixing in `unsloth/tokenizer_utils.py`

#### From Vision_Model_Finetuning:
- `FastVisionModel.from_pretrained()` - Vision model loading in `unsloth/models/loader.py`
- `FastVisionModel.get_peft_model()` - Vision LoRA setup
- `UnslothVisionDataCollator` - Multimodal batching in `unsloth/trainer.py`
- `FastVisionModel.for_training()` - Training mode setup
- `process_vision_info()` - Image preprocessing in `unsloth/models/_utils.py`

### Important Classes/Functions Used

1. **FastLanguageModel** (`unsloth/models/loader.py`)
   - Main entry point for all language model operations
   - Handles architecture detection, quantization, and patching

2. **FastLlamaModel** (`unsloth/models/llama.py`)
   - Reference implementation for model optimization
   - Contains `get_peft_model()`, attention patching, etc.

3. **FastVisionModel** (`unsloth/models/loader.py`)
   - Alias for FastModel for vision-language models

4. **FastBaseModel** (`unsloth/models/vision.py`)
   - Base class for general model handling

5. **UnslothTrainer** (`unsloth/trainer.py`)
   - Custom trainer with sample packing support

6. **Triton Kernels** (`unsloth/kernels/`)
   - `fast_lora.py` - Fused LoRA operations
   - `cross_entropy_loss.py` - Chunked loss computation
   - `rms_layernorm.py` - RMSNorm optimization
   - `rope_embedding.py` - RoPE position encoding

### Principles to Create (from Workflow Steps)

#### QLoRA_Finetuning Principles:
1. `unslothai_unsloth_Environment_Initialization` - Import order requirements
2. `unslothai_unsloth_Model_Loading` - Quantized model loading
3. `unslothai_unsloth_LoRA_Configuration` - LoRA adapter setup
4. `unslothai_unsloth_Data_Formatting` - Chat template application
5. `unslothai_unsloth_Training_Setup` - Trainer configuration
6. `unslothai_unsloth_Training_Execution` - Training loop
7. `unslothai_unsloth_Model_Saving` - Model export options

#### Model_Export_GGUF Principles:
1. `unslothai_unsloth_Export_Preparation` - Pre-export setup
2. `unslothai_unsloth_Model_Merging` - LoRA weight merging
3. `unslothai_unsloth_GGUF_Conversion` - GGUF format conversion
4. `unslothai_unsloth_Ollama_Integration` - Modelfile generation
5. `unslothai_unsloth_Hub_Upload` - HuggingFace Hub push
6. `unslothai_unsloth_Ollama_Deployment` - Local deployment

#### Vision_Model_Finetuning Principles:
1. `unslothai_unsloth_VLM_Environment_Initialization` - Vision imports
2. `unslothai_unsloth_VLM_Loading` - Vision model loading
3. `unslothai_unsloth_VLM_LoRA_Configuration` - Vision/language adapter setup
4. `unslothai_unsloth_VLM_Data_Formatting` - Multimodal data prep
5. `unslothai_unsloth_VLM_Training_Setup` - Vision trainer config
6. `unslothai_unsloth_VLM_Training_Execution` - Vision training loop
7. `unslothai_unsloth_VLM_Saving` - Vision model export

### Implementations to Create

Key implementations referenced by workflows:
- `unslothai_unsloth_FastLanguageModel`
- `unslothai_unsloth_FastVisionModel`
- `unslothai_unsloth_FastLlamaModel`
- `unslothai_unsloth_get_peft_model`
- `unslothai_unsloth_unsloth_save_model`
- `unslothai_unsloth_save_to_gguf`
- `unslothai_unsloth_merge_lora`
- `unslothai_unsloth_UnslothVisionDataCollator`
- `unslothai_unsloth_OLLAMA_TEMPLATES`
