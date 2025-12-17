# Phase 1: Anchoring Report

**Repository:** unslothai/unsloth
**Date:** 2025-12-17
**Status:** Complete

---

## Summary

- **Workflows created:** 3
- **Total steps documented:** 23
- **Implementation hints captured:** 23
- **Principles identified:** 18 unique (some shared across workflows)

---

## Workflows Created

| Workflow | Source Files | Steps | Implementation APIs |
|----------|--------------|-------|---------------------|
| `unslothai_unsloth_QLoRA_Finetuning` | loader.py, llama.py, chat_templates.py, trainer.py, save.py | 7 | FastLanguageModel.from_pretrained, get_peft_model, SFTTrainer, save_pretrained_merged |
| `unslothai_unsloth_GRPO_Reinforcement_Learning` | loader.py, llama.py, rl.py, rl_replacements.py, save.py | 8 | FastLanguageModel.from_pretrained (vLLM), GRPOTrainer, reward functions |
| `unslothai_unsloth_Model_Export` | save.py, ollama_template_mappers.py, tokenizer_utils.py | 8 | save_pretrained, save_pretrained_merged, save_pretrained_gguf, push_to_hub |

---

## Coverage Summary

### Source Files Covered

| Category | Files Covered | Total Files | Coverage % |
|----------|---------------|-------------|------------|
| Core Package | 25 | 77 | 32% |
| Test Files | 8 | 37 | 22% |
| Total | 33 | 116 | 28% |

### Key Files by Workflow

**QLoRA_Finetuning:**
- `unsloth/__init__.py` - Environment initialization
- `unsloth/models/loader.py` - Model loading orchestration
- `unsloth/models/llama.py` - LoRA injection and model patching
- `unsloth/chat_templates.py` - Data formatting
- `unsloth/trainer.py` - Training optimizations
- `unsloth/save.py` - Model saving
- `unsloth/kernels/` - Triton kernels (cross_entropy, rope, rms_layernorm, swiglu)

**GRPO_Reinforcement_Learning:**
- `unsloth/models/loader.py` - Model loading with vLLM
- `unsloth/models/rl.py` - RL trainer optimizations
- `unsloth/models/rl_replacements.py` - TRL trainer patches
- `unsloth/chat_templates.py` - Chat template setup
- `tests/saving/language_models/test_save_merged_grpo_model.py` - Example GRPO flow

**Model_Export:**
- `unsloth/save.py` - All export methods
- `unsloth/ollama_template_mappers.py` - Ollama Modelfile generation
- `unsloth/tokenizer_utils.py` - Tokenizer handling for export
- `unsloth/utils/hf_hub.py` - HuggingFace Hub utilities

---

## Implementation Context Captured

| Workflow | Principles | API Docs | Wrapper Docs | Pattern Docs |
|----------|------------|----------|--------------|--------------|
| QLoRA_Finetuning | 7 | 5 | 2 | 0 |
| GRPO_Reinforcement_Learning | 8 | 4 | 3 | 1 |
| Model_Export | 8 | 5 | 0 | 3 |
| **Total** | **23** | **14** | **5** | **4** |

---

## Notes for Excavation Phase

### APIs to Extract (with Source Locations)

| API | Source | Used By Principles | Type |
|-----|--------|-------------------|------|
| `FastLanguageModel.from_pretrained` | `loader.py:L120-620` | Model_Loading, RL_Model_Loading | API Doc |
| `FastLanguageModel.get_peft_model` | `llama.py:L2578-3100` | LoRA_Configuration | API Doc |
| `get_chat_template` | `chat_templates.py:L50-500` | Data_Formatting, Chat_Template_Setup | API Doc |
| `save_pretrained_merged` | `save.py:L200-800` | Model_Saving, Merged_Export | API Doc |
| `save_pretrained_gguf` | `save.py:L800-1500` | GGUF_Conversion | API Doc |
| `push_to_hub` | `save.py:L1500-2000` | Hub_Upload | API Doc |
| Ollama Modelfile generation | `ollama_template_mappers.py:L1-2192` | Ollama_Export | API Doc |

### External Dependencies to Document

| Library | APIs Used | Wrapper Doc Needed |
|---------|-----------|-------------------|
| TRL | SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig | Yes - document Unsloth-specific usage |
| PEFT | PeftModel, LoraConfig | Partial - mainly via get_peft_model |
| bitsandbytes | Linear4bit, quantization | Internal - handled by loader |
| vLLM | Fast inference backend | Yes - document fast_inference=True |
| llama.cpp | convert_hf_to_gguf.py, llama-quantize | Yes - document GGUF workflow |

### User-Defined Patterns to Document

| Pattern | Interface | Used By |
|---------|-----------|---------|
| Reward Function | `def reward_func(prompts, completions, **kwargs) -> List[float]` | GRPO_Training |
| Chat Template Formatting | `tokenizer.apply_chat_template(messages, ...)` | Data_Formatting |
| Export Format Selection | Decision tree for format choice | Model_Export |

---

## Unique Principles Identified

| Principle | Used By Workflows |
|-----------|-------------------|
| Environment_Initialization | QLoRA_Finetuning |
| Model_Loading | QLoRA_Finetuning |
| RL_Model_Loading | GRPO_Reinforcement_Learning |
| LoRA_Configuration | QLoRA_Finetuning, GRPO_Reinforcement_Learning |
| Data_Formatting | QLoRA_Finetuning |
| Chat_Template_Setup | GRPO_Reinforcement_Learning |
| Training_Configuration | QLoRA_Finetuning |
| SFT_Training | QLoRA_Finetuning, GRPO_Reinforcement_Learning |
| Reward_Function_Interface | GRPO_Reinforcement_Learning |
| GRPO_Configuration | GRPO_Reinforcement_Learning |
| GRPO_Training | GRPO_Reinforcement_Learning |
| Model_Saving | QLoRA_Finetuning, GRPO_Reinforcement_Learning, Model_Export |
| Training_Verification | Model_Export |
| Export_Format_Selection | Model_Export |
| LoRA_Export | Model_Export |
| Merged_Export | Model_Export |
| GGUF_Conversion | Model_Export |
| Ollama_Export | Model_Export |
| Hub_Upload | Model_Export |
| Export_Validation | Model_Export |

---

## Files Updated

| File | Changes |
|------|---------|
| `_RepoMap_unslothai_unsloth.md` | Added Coverage column with workflow references |
| `_WorkflowIndex.md` | Complete rewrite with detailed implementation context for all 3 workflows |
| `workflows/unslothai_unsloth_QLoRA_Finetuning.md` | Created (7 steps) |
| `workflows/unslothai_unsloth_GRPO_Reinforcement_Learning.md` | Created (8 steps) |
| `workflows/unslothai_unsloth_Model_Export.md` | Created (8 steps) |

---

## Recommendations for Phase 2

### High Priority Implementations

1. **FastLanguageModel.from_pretrained** - Central API, used by all workflows
2. **FastLanguageModel.get_peft_model** - Critical for LoRA setup
3. **save_pretrained_merged** - Key export functionality
4. **get_chat_template** - Essential for data formatting

### Principle Pages to Create First

1. **Model_Loading** - Foundation for all workflows
2. **LoRA_Configuration** - Core training concept
3. **Model_Saving** - Shared across workflows
4. **SFT_Training** - Most common training type

### Environment Page Needed

- **unslothai_unsloth_CUDA** - Document CUDA requirements, Triton dependencies, bitsandbytes linking

---

## Phase 1 Status: COMPLETE

- 3 Workflow pages created
- WorkflowIndex updated with comprehensive implementation context
- RepoMap updated with coverage information
- All 23 steps documented with API calls, source locations, and dependencies
