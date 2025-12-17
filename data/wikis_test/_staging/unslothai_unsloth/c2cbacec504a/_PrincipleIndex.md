# Principle Index: unslothai_unsloth

> Tracks Principle pages and their connections to Implementations, Workflows, etc.
> **Update IMMEDIATELY** after creating or modifying a Principle page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| unslothai_unsloth_Environment_Initialization | [→](./principles/unslothai_unsloth_Environment_Initialization.md) | ✅Impl:unslothai_unsloth_import_unsloth, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning | Import order for kernel patching |
| unslothai_unsloth_Model_Loading | [→](./principles/unslothai_unsloth_Model_Loading.md) | ✅Impl:unslothai_unsloth_FastLanguageModel_from_pretrained, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning | 4-bit QLoRA model loading |
| unslothai_unsloth_RL_Model_Loading | [→](./principles/unslothai_unsloth_RL_Model_Loading.md) | ✅Impl:unslothai_unsloth_FastLanguageModel_from_pretrained_vllm, ✅Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning | vLLM-enabled loading for RL |
| unslothai_unsloth_LoRA_Configuration | [→](./principles/unslothai_unsloth_LoRA_Configuration.md) | ✅Impl:unslothai_unsloth_get_peft_model, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning | Parameter-efficient training config |
| unslothai_unsloth_Data_Formatting | [→](./principles/unslothai_unsloth_Data_Formatting.md) | ✅Impl:unslothai_unsloth_get_chat_template, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning | Chat template for dataset prep |
| unslothai_unsloth_Chat_Template_Setup | [→](./principles/unslothai_unsloth_Chat_Template_Setup.md) | ✅Impl:unslothai_unsloth_get_chat_template, ✅Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning | RL-specific template config |
| unslothai_unsloth_Training_Configuration | [→](./principles/unslothai_unsloth_Training_Configuration.md) | ✅Impl:unslothai_unsloth_SFTTrainer_usage, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning | SFT trainer setup |
| unslothai_unsloth_SFT_Training | [→](./principles/unslothai_unsloth_SFT_Training.md) | ✅Impl:unslothai_unsloth_trainer_train, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning | Training execution |
| unslothai_unsloth_Model_Saving | [→](./principles/unslothai_unsloth_Model_Saving.md) | ✅Impl:unslothai_unsloth_save_pretrained_merged, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning | Model checkpoint saving |
| unslothai_unsloth_Reward_Function_Interface | [→](./principles/unslothai_unsloth_Reward_Function_Interface.md) | ✅Impl:unslothai_unsloth_reward_function_pattern, ✅Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning | Reward function contract |
| unslothai_unsloth_GRPO_Configuration | [→](./principles/unslothai_unsloth_GRPO_Configuration.md) | ✅Impl:unslothai_unsloth_GRPOConfig, ✅Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning | GRPO hyperparameter setup |
| unslothai_unsloth_GRPO_Training | [→](./principles/unslothai_unsloth_GRPO_Training.md) | ✅Impl:unslothai_unsloth_GRPOTrainer_train, ✅Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning | RL training execution |
| unslothai_unsloth_Training_Verification | [→](./principles/unslothai_unsloth_Training_Verification.md) | ✅Impl:unslothai_unsloth_model_generate, ✅Workflow:unslothai_unsloth_Model_Export | Pre-export quality check |
| unslothai_unsloth_Export_Format_Selection | [→](./principles/unslothai_unsloth_Export_Format_Selection.md) | ✅Impl:unslothai_unsloth_export_format_selection_pattern, ✅Workflow:unslothai_unsloth_Model_Export | Format decision criteria |
| unslothai_unsloth_LoRA_Export | [→](./principles/unslothai_unsloth_LoRA_Export.md) | ✅Impl:unslothai_unsloth_save_pretrained_lora, ✅Workflow:unslothai_unsloth_Model_Export | Adapter-only export |
| unslothai_unsloth_Merged_Export | [→](./principles/unslothai_unsloth_Merged_Export.md) | ✅Impl:unslothai_unsloth_save_pretrained_merged, ✅Workflow:unslothai_unsloth_Model_Export | Merged weight export |
| unslothai_unsloth_GGUF_Conversion | [→](./principles/unslothai_unsloth_GGUF_Conversion.md) | ✅Impl:unslothai_unsloth_save_pretrained_gguf, ✅Workflow:unslothai_unsloth_Model_Export | llama.cpp format conversion |
| unslothai_unsloth_Ollama_Export | [→](./principles/unslothai_unsloth_Ollama_Export.md) | ✅Impl:unslothai_unsloth_ollama_modelfile, ✅Workflow:unslothai_unsloth_Model_Export | Ollama deployment packaging |
| unslothai_unsloth_Hub_Upload | [→](./principles/unslothai_unsloth_Hub_Upload.md) | ✅Impl:unslothai_unsloth_push_to_hub, ✅Workflow:unslothai_unsloth_Model_Export | HuggingFace Hub publishing |
| unslothai_unsloth_Export_Validation | [→](./principles/unslothai_unsloth_Export_Validation.md) | ✅Impl:unslothai_unsloth_load_and_validate, ✅Workflow:unslothai_unsloth_Model_Export | Post-export verification |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
