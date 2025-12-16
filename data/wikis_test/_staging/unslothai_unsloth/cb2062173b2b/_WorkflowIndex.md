# Workflow Index: unslothai_unsloth

> Tracks Workflow pages and their connections to Implementations, Principles, etc.
> **Update IMMEDIATELY** after creating or modifying a Workflow page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| unslothai_unsloth_QLoRA_Finetuning | [→](./workflows/unslothai_unsloth_QLoRA_Finetuning.md) | ✅Principle:unslothai_unsloth_Environment_Setup, ✅Principle:unslothai_unsloth_Model_Loading, ✅Principle:unslothai_unsloth_LoRA_Configuration, ✅Principle:unslothai_unsloth_Data_Formatting, ✅Principle:unslothai_unsloth_SFT_Training, ✅Principle:unslothai_unsloth_Model_Export, ✅Impl:unslothai_unsloth_FastLanguageModel, ✅Impl:unslothai_unsloth_get_peft_model, ✅Impl:unslothai_unsloth_get_chat_template, ✅Impl:unslothai_unsloth_train_on_responses_only, ✅Impl:unslothai_unsloth_save_pretrained_merged, ✅Heuristic:unslothai_unsloth_LoRA_Rank_Selection, ✅Heuristic:unslothai_unsloth_Gradient_Checkpointing, ✅Heuristic:unslothai_unsloth_Memory_Management, ✅Heuristic:unslothai_unsloth_Sample_Packing | Main QLoRA fine-tuning workflow with 4-bit quantization |
| unslothai_unsloth_GGUF_Export | [→](./workflows/unslothai_unsloth_GGUF_Export.md) | ✅Principle:unslothai_unsloth_Model_Loading, ✅Principle:unslothai_unsloth_LoRA_Merging, ✅Principle:unslothai_unsloth_GGUF_Conversion, ✅Principle:unslothai_unsloth_Ollama_Integration, ✅Principle:unslothai_unsloth_Model_Deployment, ✅Impl:unslothai_unsloth_FastLanguageModel, ✅Impl:unslothai_unsloth_save_pretrained_gguf, ✅Heuristic:unslothai_unsloth_Quantization_Method_Selection | Export to GGUF for llama.cpp/Ollama deployment |
| unslothai_unsloth_GRPO_Training | [→](./workflows/unslothai_unsloth_GRPO_Training.md) | ✅Principle:unslothai_unsloth_Model_Loading, ✅Principle:unslothai_unsloth_LoRA_Configuration, ✅Principle:unslothai_unsloth_Data_Formatting, ✅Principle:unslothai_unsloth_Reward_Functions, ✅Principle:unslothai_unsloth_GRPO_Training, ✅Principle:unslothai_unsloth_Model_Export, ✅Impl:unslothai_unsloth_FastLanguageModel, ✅Impl:unslothai_unsloth_get_peft_model, ✅Impl:unslothai_unsloth_get_chat_template, ✅Impl:unslothai_unsloth_save_pretrained_merged, ✅Env:unslothai_unsloth_vLLM, ✅Heuristic:unslothai_unsloth_RL_Learning_Rate | RL training with GRPO for reasoning models |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
