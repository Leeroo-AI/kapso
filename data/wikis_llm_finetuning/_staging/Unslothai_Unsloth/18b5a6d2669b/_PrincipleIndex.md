# Principle Index: Unslothai_Unsloth

> Tracks Principle pages and their connections to Implementations, Workflows, etc.
> **Update IMMEDIATELY** after creating or modifying a Principle page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| Unslothai_Unsloth_Model_Loading | [→](./principles/Unslothai_Unsloth_Model_Loading.md) | ✅Impl:Unslothai_Unsloth_FastLanguageModel_from_pretrained, ✅Workflow:Unslothai_Unsloth_QLoRA_Finetuning, ✅Workflow:Unslothai_Unsloth_CLI_Finetuning | 4-bit model loading |
| Unslothai_Unsloth_LoRA_Configuration | [→](./principles/Unslothai_Unsloth_LoRA_Configuration.md) | ✅Impl:Unslothai_Unsloth_get_peft_model, ✅Workflow:Unslothai_Unsloth_QLoRA_Finetuning, ✅Workflow:Unslothai_Unsloth_CLI_Finetuning | LoRA adapter setup |
| Unslothai_Unsloth_Data_Formatting | [→](./principles/Unslothai_Unsloth_Data_Formatting.md) | ✅Impl:Unslothai_Unsloth_get_chat_template, ✅Workflow:Unslothai_Unsloth_QLoRA_Finetuning | Chat template application |
| Unslothai_Unsloth_Training_Configuration | [→](./principles/Unslothai_Unsloth_Training_Configuration.md) | ✅Impl:Unslothai_Unsloth_UnslothTrainingArguments, ✅Workflow:Unslothai_Unsloth_QLoRA_Finetuning, ✅Workflow:Unslothai_Unsloth_CLI_Finetuning | Training hyperparameters |
| Unslothai_Unsloth_Supervised_Finetuning | [→](./principles/Unslothai_Unsloth_Supervised_Finetuning.md) | ✅Impl:Unslothai_Unsloth_SFTTrainer_train, ✅Workflow:Unslothai_Unsloth_QLoRA_Finetuning, ✅Workflow:Unslothai_Unsloth_CLI_Finetuning | SFT execution |
| Unslothai_Unsloth_Model_Saving | [→](./principles/Unslothai_Unsloth_Model_Saving.md) | ✅Impl:Unslothai_Unsloth_save_pretrained, ✅Workflow:Unslothai_Unsloth_QLoRA_Finetuning, ✅Workflow:Unslothai_Unsloth_GRPO_Training, ✅Workflow:Unslothai_Unsloth_CLI_Finetuning | Model serialization |
| Unslothai_Unsloth_RL_Model_Loading | [→](./principles/Unslothai_Unsloth_RL_Model_Loading.md) | ✅Impl:Unslothai_Unsloth_FastLanguageModel_from_pretrained_vllm, ✅Workflow:Unslothai_Unsloth_GRPO_Training | vLLM-enabled loading |
| Unslothai_Unsloth_RL_LoRA_Configuration | [→](./principles/Unslothai_Unsloth_RL_LoRA_Configuration.md) | ✅Impl:Unslothai_Unsloth_get_peft_model_rl, ✅Workflow:Unslothai_Unsloth_GRPO_Training | LoRA for RL |
| Unslothai_Unsloth_Chat_Template_Configuration | [→](./principles/Unslothai_Unsloth_Chat_Template_Configuration.md) | ✅Impl:Unslothai_Unsloth_get_chat_template, ✅Workflow:Unslothai_Unsloth_GRPO_Training | Chat template for RL |
| Unslothai_Unsloth_RL_Dataset_Preparation | [→](./principles/Unslothai_Unsloth_RL_Dataset_Preparation.md) | ✅Impl:Unslothai_Unsloth_dataset_mapping_pattern, ✅Workflow:Unslothai_Unsloth_GRPO_Training | Prompt dataset formatting |
| Unslothai_Unsloth_Reward_Functions | [→](./principles/Unslothai_Unsloth_Reward_Functions.md) | ✅Impl:Unslothai_Unsloth_reward_function_pattern, ✅Workflow:Unslothai_Unsloth_GRPO_Training | Reward function design |
| Unslothai_Unsloth_SFT_Pretraining | [→](./principles/Unslothai_Unsloth_SFT_Pretraining.md) | ✅Impl:Unslothai_Unsloth_train_on_responses_only, ✅Workflow:Unslothai_Unsloth_GRPO_Training | Optional SFT phase |
| Unslothai_Unsloth_GRPO_Configuration | [→](./principles/Unslothai_Unsloth_GRPO_Configuration.md) | ✅Impl:Unslothai_Unsloth_UnslothGRPOConfig, ✅Workflow:Unslothai_Unsloth_GRPO_Training | GRPO hyperparameters |
| Unslothai_Unsloth_GRPO_Execution | [→](./principles/Unslothai_Unsloth_GRPO_Execution.md) | ✅Impl:Unslothai_Unsloth_UnslothGRPOTrainer, ✅Workflow:Unslothai_Unsloth_GRPO_Training | GRPO training loop |
| Unslothai_Unsloth_Vision_Model_Loading | [→](./principles/Unslothai_Unsloth_Vision_Model_Loading.md) | ✅Impl:Unslothai_Unsloth_FastVisionModel_from_pretrained, ✅Workflow:Unslothai_Unsloth_Vision_Finetuning | VLM loading |
| Unslothai_Unsloth_Vision_LoRA_Configuration | [→](./principles/Unslothai_Unsloth_Vision_LoRA_Configuration.md) | ✅Impl:Unslothai_Unsloth_FastBaseModel_get_peft_model, ✅Workflow:Unslothai_Unsloth_Vision_Finetuning | Vision/language LoRA |
| Unslothai_Unsloth_Multimodal_Data_Preparation | [→](./principles/Unslothai_Unsloth_Multimodal_Data_Preparation.md) | ✅Impl:Unslothai_Unsloth_multimodal_dataset_pattern, ✅Workflow:Unslothai_Unsloth_Vision_Finetuning | Image-text datasets |
| Unslothai_Unsloth_Vision_Training_Mode | [→](./principles/Unslothai_Unsloth_Vision_Training_Mode.md) | ✅Impl:Unslothai_Unsloth_FastBaseModel_for_training, ✅Workflow:Unslothai_Unsloth_Vision_Finetuning | Training mode config |
| Unslothai_Unsloth_Vision_Training | [→](./principles/Unslothai_Unsloth_Vision_Training.md) | ✅Impl:Unslothai_Unsloth_SFTTrainer_vision, ✅Workflow:Unslothai_Unsloth_Vision_Finetuning | VLM training |
| Unslothai_Unsloth_Vision_Inference_Mode | [→](./principles/Unslothai_Unsloth_Vision_Inference_Mode.md) | ✅Impl:Unslothai_Unsloth_FastBaseModel_for_inference, ✅Workflow:Unslothai_Unsloth_Vision_Finetuning | Inference mode config |
| Unslothai_Unsloth_Vision_Model_Saving | [→](./principles/Unslothai_Unsloth_Vision_Model_Saving.md) | ✅Impl:Unslothai_Unsloth_save_pretrained_vision, ✅Workflow:Unslothai_Unsloth_Vision_Finetuning | VLM serialization |
| Unslothai_Unsloth_Model_Preparation | [→](./principles/Unslothai_Unsloth_Model_Preparation.md) | ✅Impl:Unslothai_Unsloth_unsloth_save_model_merged, ✅Workflow:Unslothai_Unsloth_GGUF_Export | LoRA merging |
| Unslothai_Unsloth_Quantization_Selection | [→](./principles/Unslothai_Unsloth_Quantization_Selection.md) | ✅Impl:Unslothai_Unsloth_ALLOWED_QUANTS, ✅Workflow:Unslothai_Unsloth_GGUF_Export | Quant method choice |
| Unslothai_Unsloth_GGUF_Export | [→](./principles/Unslothai_Unsloth_GGUF_Export.md) | ✅Impl:Unslothai_Unsloth_save_to_gguf, ✅Workflow:Unslothai_Unsloth_GGUF_Export | GGUF conversion |
| Unslothai_Unsloth_Ollama_Modelfile | [→](./principles/Unslothai_Unsloth_Ollama_Modelfile.md) | ✅Impl:Unslothai_Unsloth_OLLAMA_TEMPLATES, ✅Workflow:Unslothai_Unsloth_GGUF_Export | Modelfile generation |
| Unslothai_Unsloth_GGUF_Hub_Upload | [→](./principles/Unslothai_Unsloth_GGUF_Hub_Upload.md) | ✅Impl:Unslothai_Unsloth_push_to_hub_gguf, ✅Workflow:Unslothai_Unsloth_GGUF_Export | Hub upload |
| Unslothai_Unsloth_GGUF_Verification | [→](./principles/Unslothai_Unsloth_GGUF_Verification.md) | ✅Impl:Unslothai_Unsloth_llama_cli_validation, ✅Workflow:Unslothai_Unsloth_GGUF_Export | Validation |
| Unslothai_Unsloth_CLI_Data_Loading | [→](./principles/Unslothai_Unsloth_CLI_Data_Loading.md) | ✅Impl:Unslothai_Unsloth_CLI, ✅Impl:Unslothai_Unsloth_RawTextDataLoader, ✅Workflow:Unslothai_Unsloth_CLI_Finetuning | Smart dataset loading |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
