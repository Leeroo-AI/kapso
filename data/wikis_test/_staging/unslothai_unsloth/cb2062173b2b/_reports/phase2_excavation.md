# Phase 2: Excavation Report

## Summary

Traced imports from Workflow pages to source code implementations. Created 6 Implementation wiki pages for the main user-facing APIs in the Unsloth library.

## Implementations Created

| Implementation | Source File | Lines | Description |
|----------------|-------------|-------|-------------|
| unslothai_unsloth_FastLanguageModel | unsloth/models/loader.py:L120-620 | ~500 | Main model loading API with 4-bit/8-bit/16-bit quantization |
| unslothai_unsloth_get_peft_model | unsloth/models/llama.py:L2578-2800 | ~220 | LoRA adapter application for PEFT fine-tuning |
| unslothai_unsloth_save_pretrained_merged | unsloth/save.py:L2653-2693, L228-506 | ~320 | Merge LoRA and save to HuggingFace format |
| unslothai_unsloth_save_pretrained_gguf | unsloth/save.py:L1776-2000 | ~225 | GGUF conversion for llama.cpp/Ollama deployment |
| unslothai_unsloth_get_chat_template | unsloth/chat_templates.py:L2123-2400 | ~280 | Chat template configuration for instruction tuning |
| unslothai_unsloth_train_on_responses_only | unsloth/chat_templates.py:L40 (re-export) | — | Response-only loss masking for SFT |

## API Coverage

- **Classes documented:** 1 (FastLanguageModel)
- **Functions documented:** 5 (get_peft_model, save_pretrained_merged, save_pretrained_gguf, get_chat_template, train_on_responses_only)
- **Total Implementation pages:** 6
- **Source files covered:** 4 (loader.py, llama.py, save.py, chat_templates.py)

## Implementation-Workflow Links

| Implementation | Workflows Covered |
|----------------|-------------------|
| FastLanguageModel | QLoRA_Finetuning, GGUF_Export, GRPO_Training |
| get_peft_model | QLoRA_Finetuning, GRPO_Training |
| save_pretrained_merged | QLoRA_Finetuning, GRPO_Training |
| save_pretrained_gguf | GGUF_Export |
| get_chat_template | QLoRA_Finetuning, GRPO_Training |
| train_on_responses_only | QLoRA_Finetuning |

## Index Updates

- **_ImplementationIndex.md**: Created with all 6 Implementation pages
- **_WorkflowIndex.md**: Updated all 3 Workflows with Implementation links (✅Impl:...)
- **_RepoMap_unslothai_unsloth.md**: Updated Coverage column for source files

## Notes for Synthesis Phase

### Concepts that need Principle pages

1. **Model_Loading** - How Unsloth auto-detects and patches model architectures
2. **LoRA_Configuration** - LoRA rank selection, target modules, alpha/dropout settings
3. **Data_Formatting** - Chat templates, ShareGPT format, tokenizer configuration
4. **SFT_Training** - Supervised fine-tuning with response-only loss
5. **GRPO_Training** - Group Relative Policy Optimization for RL
6. **GGUF_Conversion** - llama.cpp quantization methods and trade-offs
7. **Model_Export** - Merging strategies (16-bit vs 4-bit)
8. **Reward_Functions** - GRPO reward model integration
9. **Environment_Setup** - CUDA, vLLM, bitsandbytes requirements

### Patterns observed across implementations

1. **Dynamic method patching**: Unsloth attaches methods (save_pretrained_merged, save_pretrained_gguf) to models at load time via `patch_saving_functions()`
2. **Architecture auto-detection**: `FastLanguageModel.from_pretrained()` routes to architecture-specific implementations (FastLlamaModel, FastMistralModel, etc.)
3. **Memory-efficient processing**: All save functions support `maximum_memory_usage` parameter for constrained systems
4. **HuggingFace integration**: Deep integration with transformers, PEFT, and huggingface_hub libraries
5. **Triton kernel optimization**: Custom Triton kernels for attention, normalization, and activation functions

### External dependencies (unsloth_zoo)

The `train_on_responses_only` function is actually implemented in `unsloth_zoo.dataset_utils` and re-exported through `unsloth.chat_templates`. This pattern appears for several training utilities.

## File Statistics

| Metric | Count |
|--------|-------|
| Implementation pages created | 6 |
| Source files with Implementation coverage | 4 |
| Total source lines documented | ~1,545 |
| Workflow connections established | 12 |
