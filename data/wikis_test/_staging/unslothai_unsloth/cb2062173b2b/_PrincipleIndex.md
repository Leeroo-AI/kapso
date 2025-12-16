# Principle Index: unslothai_unsloth

> Tracks Principle pages and their connections to Implementations, Workflows, etc.
> **Update IMMEDIATELY** after creating or modifying a Principle page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| unslothai_unsloth_Model_Loading | [→](./principles/unslothai_unsloth_Model_Loading.md) | ✅Impl:unslothai_unsloth_FastLanguageModel, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Workflow:unslothai_unsloth_GGUF_Export, ✅Workflow:unslothai_unsloth_GRPO_Training | 4-bit NF4 quantized model loading |
| unslothai_unsloth_LoRA_Configuration | [→](./principles/unslothai_unsloth_LoRA_Configuration.md) | ✅Impl:unslothai_unsloth_get_peft_model, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Workflow:unslothai_unsloth_GRPO_Training, ✅Heuristic:unslothai_unsloth_LoRA_Rank_Selection, ✅Heuristic:unslothai_unsloth_LoRA_Dropout_Bias | Low-Rank Adaptation parameter-efficient fine-tuning |
| unslothai_unsloth_Data_Formatting | [→](./principles/unslothai_unsloth_Data_Formatting.md) | ✅Impl:unslothai_unsloth_get_chat_template, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Workflow:unslothai_unsloth_GRPO_Training | Chat templates and instruction formatting |
| unslothai_unsloth_SFT_Training | [→](./principles/unslothai_unsloth_SFT_Training.md) | ✅Impl:unslothai_unsloth_train_on_responses_only, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Heuristic:unslothai_unsloth_Sample_Packing | Supervised fine-tuning with response-only loss |
| unslothai_unsloth_Model_Export | [→](./principles/unslothai_unsloth_Model_Export.md) | ✅Impl:unslothai_unsloth_save_pretrained_merged, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Workflow:unslothai_unsloth_GRPO_Training | LoRA merging and HuggingFace format export |
| unslothai_unsloth_GGUF_Conversion | [→](./principles/unslothai_unsloth_GGUF_Conversion.md) | ✅Impl:unslothai_unsloth_save_pretrained_gguf, ✅Workflow:unslothai_unsloth_GGUF_Export, ✅Heuristic:unslothai_unsloth_Quantization_Method_Selection | GGUF format quantization for llama.cpp |
| unslothai_unsloth_GRPO_Training | [→](./principles/unslothai_unsloth_GRPO_Training.md) | ✅Impl:unslothai_unsloth_FastLanguageModel, ✅Impl:unslothai_unsloth_get_peft_model, ✅Workflow:unslothai_unsloth_GRPO_Training, ✅Heuristic:unslothai_unsloth_RL_Learning_Rate | Group Relative Policy Optimization for reasoning |
| unslothai_unsloth_Reward_Functions | [→](./principles/unslothai_unsloth_Reward_Functions.md) | ✅Impl:unslothai_unsloth_FastLanguageModel, ✅Workflow:unslothai_unsloth_GRPO_Training | RL reward function design patterns |
| unslothai_unsloth_LoRA_Merging | [→](./principles/unslothai_unsloth_LoRA_Merging.md) | ✅Impl:unslothai_unsloth_save_pretrained_merged, ✅Impl:unslothai_unsloth_save_pretrained_gguf, ✅Workflow:unslothai_unsloth_GGUF_Export | Mathematical LoRA merge operation |
| unslothai_unsloth_Environment_Setup | [→](./principles/unslothai_unsloth_Environment_Setup.md) | ✅Impl:unslothai_unsloth_FastLanguageModel, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Heuristic:unslothai_unsloth_Gradient_Checkpointing | Hardware detection and environment configuration |
| unslothai_unsloth_Ollama_Integration | [→](./principles/unslothai_unsloth_Ollama_Integration.md) | ✅Impl:unslothai_unsloth_save_pretrained_gguf, ✅Workflow:unslothai_unsloth_GGUF_Export | Ollama Modelfile generation |
| unslothai_unsloth_Model_Deployment | [→](./principles/unslothai_unsloth_Model_Deployment.md) | ✅Impl:unslothai_unsloth_save_pretrained_merged, ✅Impl:unslothai_unsloth_save_pretrained_gguf, ✅Workflow:unslothai_unsloth_GGUF_Export | Production deployment strategies |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
