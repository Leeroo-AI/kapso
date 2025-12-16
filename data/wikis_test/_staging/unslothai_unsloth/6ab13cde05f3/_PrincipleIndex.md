# Principle Index: unslothai_unsloth

> Tracks Principle pages and their connections to Implementations, Workflows, etc.
> **Update IMMEDIATELY** after creating or modifying a Principle page.

## Pages

| Page | File | Implementations | Notes |
|------|------|-----------------|-------|
| unslothai_unsloth_Model_Loading | [→](./principles/unslothai_unsloth_Model_Loading.md) | ✅Impl:unslothai_unsloth_FastLanguageModel, ✅Impl:unslothai_unsloth_FastVisionModel | 4-bit NF4 quantization for model loading |
| unslothai_unsloth_LoRA_Injection | [→](./principles/unslothai_unsloth_LoRA_Injection.md) | ✅Impl:unslothai_unsloth_get_peft_model | Low-rank adaptation with fused kernels |
| unslothai_unsloth_Weight_Merging | [→](./principles/unslothai_unsloth_Weight_Merging.md) | ✅Impl:unslothai_unsloth_save_pretrained_merged, ✅Impl:unslothai_unsloth_save_pretrained_gguf | LoRA weight fusion for deployment |
| unslothai_unsloth_GGUF_Conversion | [→](./principles/unslothai_unsloth_GGUF_Conversion.md) | ✅Impl:unslothai_unsloth_save_pretrained_gguf | GGUF format export with quantization |
| unslothai_unsloth_Vision_Model_Loading | [→](./principles/unslothai_unsloth_Vision_Model_Loading.md) | ✅Impl:unslothai_unsloth_FastVisionModel | VLM loading with selective quantization |
| unslothai_unsloth_RL_Setup | [→](./principles/unslothai_unsloth_RL_Setup.md) | ✅Impl:unslothai_unsloth_PatchFastRL | TRL trainer optimization for RL |
| unslothai_unsloth_Package_Initialization | [→](./principles/unslothai_unsloth_Package_Initialization.md) | ✅Impl:unslothai_unsloth_FastLanguageModel | Import order and auto-patching |
| unslothai_unsloth_Data_Formatting | [→](./principles/unslothai_unsloth_Data_Formatting.md) | ✅Impl:unslothai_unsloth_FastLanguageModel | Chat template application |
| unslothai_unsloth_SFT_Training | [→](./principles/unslothai_unsloth_SFT_Training.md) | ✅Impl:unslothai_unsloth_FastLanguageModel | Supervised fine-tuning configuration |
| unslothai_unsloth_Model_Saving | [→](./principles/unslothai_unsloth_Model_Saving.md) | ✅Impl:unslothai_unsloth_save_pretrained_merged | Model persistence options |
| unslothai_unsloth_Vision_LoRA_Injection | [→](./principles/unslothai_unsloth_Vision_LoRA_Injection.md) | ✅Impl:unslothai_unsloth_FastVisionModel | VLM component-aware LoRA |
| unslothai_unsloth_Vision_Data_Formatting | [→](./principles/unslothai_unsloth_Vision_Data_Formatting.md) | ✅Impl:unslothai_unsloth_FastVisionModel | Image-text data preparation |
| unslothai_unsloth_Vision_SFT_Training | [→](./principles/unslothai_unsloth_Vision_SFT_Training.md) | ✅Impl:unslothai_unsloth_FastVisionModel | VLM training configuration |
| unslothai_unsloth_Vision_Model_Saving | [→](./principles/unslothai_unsloth_Vision_Model_Saving.md) | ✅Impl:unslothai_unsloth_save_pretrained_merged | VLM export options |
| unslothai_unsloth_Model_Preparation | [→](./principles/unslothai_unsloth_Model_Preparation.md) | ✅Impl:unslothai_unsloth_save_pretrained_merged | Pre-export model setup |
| unslothai_unsloth_GGUF_Validation | [→](./principles/unslothai_unsloth_GGUF_Validation.md) | ✅Impl:unslothai_unsloth_save_pretrained_gguf | GGUF file verification |
| unslothai_unsloth_Hub_Upload | [→](./principles/unslothai_unsloth_Hub_Upload.md) | ✅Impl:unslothai_unsloth_save_pretrained_merged | HuggingFace Hub deployment |
| unslothai_unsloth_Ollama_Integration | [→](./principles/unslothai_unsloth_Ollama_Integration.md) | ✅Impl:unslothai_unsloth_save_pretrained_gguf | Ollama deployment |
| unslothai_unsloth_RL_Model_Loading | [→](./principles/unslothai_unsloth_RL_Model_Loading.md) | ✅Impl:unslothai_unsloth_PatchFastRL | vLLM-enabled model loading |
| unslothai_unsloth_RL_LoRA_Setup | [→](./principles/unslothai_unsloth_RL_LoRA_Setup.md) | ✅Impl:unslothai_unsloth_PatchFastRL | High-rank LoRA for RL |
| unslothai_unsloth_RL_Data_Preparation | [→](./principles/unslothai_unsloth_RL_Data_Preparation.md) | ✅Impl:unslothai_unsloth_PatchFastRL | Prompt-only datasets |
| unslothai_unsloth_Reward_Definition | [→](./principles/unslothai_unsloth_Reward_Definition.md) | ✅Impl:unslothai_unsloth_PatchFastRL | Reward function design |
| unslothai_unsloth_GRPO_Configuration | [→](./principles/unslothai_unsloth_GRPO_Configuration.md) | ✅Impl:unslothai_unsloth_PatchFastRL | GRPO hyperparameters |
| unslothai_unsloth_GRPO_Training | [→](./principles/unslothai_unsloth_GRPO_Training.md) | ✅Impl:unslothai_unsloth_PatchFastRL | GRPO training loop |
| unslothai_unsloth_RL_Model_Saving | [→](./principles/unslothai_unsloth_RL_Model_Saving.md) | ✅Impl:unslothai_unsloth_save_pretrained_merged | RL model export |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
