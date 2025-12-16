# Implementation Index: unslothai_unsloth

> Tracks Implementation pages and their connections to Principles, Environments, etc.
> **Update IMMEDIATELY** after creating or modifying a Implementation page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| unslothai_unsloth_FastLanguageModel | [→](./implementations/unslothai_unsloth_FastLanguageModel.md) | ✅Principle:unslothai_unsloth_Model_Loading, ✅Principle:unslothai_unsloth_Environment_Setup, ✅Env:unslothai_unsloth_CUDA_Compute, ✅Env:unslothai_unsloth_vLLM, ✅Heuristic:unslothai_unsloth_Gradient_Checkpointing, ✅Heuristic:unslothai_unsloth_Memory_Management, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Workflow:unslothai_unsloth_GGUF_Export, ✅Workflow:unslothai_unsloth_GRPO_Training | loader.py:L120-620 - Main model loading API |
| unslothai_unsloth_get_peft_model | [→](./implementations/unslothai_unsloth_get_peft_model.md) | ✅Principle:unslothai_unsloth_LoRA_Configuration, ✅Env:unslothai_unsloth_CUDA_Compute, ✅Heuristic:unslothai_unsloth_LoRA_Rank_Selection, ✅Heuristic:unslothai_unsloth_LoRA_Dropout_Bias, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Workflow:unslothai_unsloth_GRPO_Training | llama.py:L2578-2800 - LoRA adapter application |
| unslothai_unsloth_save_pretrained_merged | [→](./implementations/unslothai_unsloth_save_pretrained_merged.md) | ✅Principle:unslothai_unsloth_Model_Export, ✅Principle:unslothai_unsloth_LoRA_Merging, ✅Env:unslothai_unsloth_Storage, ✅Heuristic:unslothai_unsloth_Memory_Management, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Workflow:unslothai_unsloth_GRPO_Training | save.py:L2653-2693 - Merge LoRA and save HF format |
| unslothai_unsloth_save_pretrained_gguf | [→](./implementations/unslothai_unsloth_save_pretrained_gguf.md) | ✅Principle:unslothai_unsloth_GGUF_Conversion, ✅Principle:unslothai_unsloth_LoRA_Merging, ✅Principle:unslothai_unsloth_Ollama_Integration, ✅Env:unslothai_unsloth_llama_cpp, ✅Heuristic:unslothai_unsloth_Quantization_Method_Selection, ✅Workflow:unslothai_unsloth_GGUF_Export | save.py:L1776-2000 - GGUF conversion and quantization |
| unslothai_unsloth_get_chat_template | [→](./implementations/unslothai_unsloth_get_chat_template.md) | ✅Principle:unslothai_unsloth_Data_Formatting, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Workflow:unslothai_unsloth_GRPO_Training | chat_templates.py:L2123-2400 - Chat template configuration |
| unslothai_unsloth_train_on_responses_only | [→](./implementations/unslothai_unsloth_train_on_responses_only.md) | ✅Principle:unslothai_unsloth_SFT_Training, ✅Heuristic:unslothai_unsloth_Sample_Packing, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning | chat_templates.py:L40 - Response-only loss masking |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
