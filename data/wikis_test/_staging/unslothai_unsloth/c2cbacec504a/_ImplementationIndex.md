# Implementation Index: unslothai_unsloth

> Tracks Implementation pages and their connections to Principles, Environments, etc.
> **Update IMMEDIATELY** after creating or modifying a Implementation page.

## Pages

| Page | File | Connections | Source | Notes |
|------|------|-------------|--------|-------|
| unslothai_unsloth_import_unsloth | [→](./implementations/unslothai_unsloth_import_unsloth.md) | ✅Principle:unslothai_unsloth_Environment_Initialization, ✅Env:unslothai_unsloth_CUDA, ✅Heuristic:unslothai_unsloth_Import_Order | `__init__.py:L1-100` | Import patching pattern |
| unslothai_unsloth_FastLanguageModel_from_pretrained | [→](./implementations/unslothai_unsloth_FastLanguageModel_from_pretrained.md) | ✅Principle:unslothai_unsloth_Model_Loading, ✅Env:unslothai_unsloth_CUDA, ✅Heuristic:unslothai_unsloth_Gradient_Checkpointing, ✅Heuristic:unslothai_unsloth_AMD_GPU_Limitations, ✅Heuristic:unslothai_unsloth_Flash_Attention_Gemma2 | `loader.py:L120-620` | QLoRA model loading |
| unslothai_unsloth_FastLanguageModel_from_pretrained_vllm | [→](./implementations/unslothai_unsloth_FastLanguageModel_from_pretrained_vllm.md) | ✅Principle:unslothai_unsloth_RL_Model_Loading, ✅Env:unslothai_unsloth_CUDA | `loader.py:L120-620` | vLLM-enabled loading for RL |
| unslothai_unsloth_get_peft_model | [→](./implementations/unslothai_unsloth_get_peft_model.md) | ✅Principle:unslothai_unsloth_LoRA_Configuration, ✅Env:unslothai_unsloth_CUDA, ✅Heuristic:unslothai_unsloth_Gradient_Checkpointing | `llama.py:L2578-3100` | LoRA adapter injection |
| unslothai_unsloth_get_chat_template | [→](./implementations/unslothai_unsloth_get_chat_template.md) | ✅Principle:unslothai_unsloth_Data_Formatting, ✅Principle:unslothai_unsloth_Chat_Template_Setup, ✅Env:unslothai_unsloth_CUDA | `chat_templates.py:L50-500` | Chat template configuration |
| unslothai_unsloth_SFTTrainer_usage | [→](./implementations/unslothai_unsloth_SFTTrainer_usage.md) | ✅Principle:unslothai_unsloth_Training_Configuration, ✅Env:unslothai_unsloth_CUDA, ✅Heuristic:unslothai_unsloth_Sample_Packing | TRL (external) | Wrapper for TRL SFTTrainer |
| unslothai_unsloth_trainer_train | [→](./implementations/unslothai_unsloth_trainer_train.md) | ✅Principle:unslothai_unsloth_SFT_Training, ✅Env:unslothai_unsloth_CUDA, ✅Heuristic:unslothai_unsloth_Sample_Packing | `trainer.py:L100-437` | Training execution |
| unslothai_unsloth_save_pretrained_merged | [→](./implementations/unslothai_unsloth_save_pretrained_merged.md) | ✅Principle:unslothai_unsloth_Model_Saving, ✅Principle:unslothai_unsloth_Merged_Export, ✅Env:unslothai_unsloth_CUDA | `save.py:L200-800` | Merged model export |
| unslothai_unsloth_reward_function_pattern | [→](./implementations/unslothai_unsloth_reward_function_pattern.md) | ✅Principle:unslothai_unsloth_Reward_Function_Interface, ✅Env:unslothai_unsloth_CUDA | User code (pattern) | Reward function interface |
| unslothai_unsloth_GRPOConfig | [→](./implementations/unslothai_unsloth_GRPOConfig.md) | ✅Principle:unslothai_unsloth_GRPO_Configuration, ✅Env:unslothai_unsloth_CUDA | TRL (external) | GRPO hyperparameter config |
| unslothai_unsloth_GRPOTrainer_train | [→](./implementations/unslothai_unsloth_GRPOTrainer_train.md) | ✅Principle:unslothai_unsloth_GRPO_Training, ✅Env:unslothai_unsloth_CUDA | `rl.py:L500-1349` | GRPO training execution |
| unslothai_unsloth_save_pretrained_gguf | [→](./implementations/unslothai_unsloth_save_pretrained_gguf.md) | ✅Principle:unslothai_unsloth_GGUF_Conversion, ✅Env:unslothai_unsloth_CUDA | `save.py:L800-1500` | GGUF quantization export |
| unslothai_unsloth_push_to_hub | [→](./implementations/unslothai_unsloth_push_to_hub.md) | ✅Principle:unslothai_unsloth_Hub_Upload, ✅Env:unslothai_unsloth_CUDA | `save.py:L1500-2000` | HuggingFace Hub upload |
| unslothai_unsloth_model_generate | [→](./implementations/unslothai_unsloth_model_generate.md) | ✅Principle:unslothai_unsloth_Training_Verification, ✅Env:unslothai_unsloth_CUDA | `llama.py:L2500-2550` | Inference verification |
| unslothai_unsloth_export_format_selection_pattern | [→](./implementations/unslothai_unsloth_export_format_selection_pattern.md) | ✅Principle:unslothai_unsloth_Export_Format_Selection | N/A (pattern) | Format decision guide |
| unslothai_unsloth_save_pretrained_lora | [→](./implementations/unslothai_unsloth_save_pretrained_lora.md) | ✅Principle:unslothai_unsloth_LoRA_Export, ✅Env:unslothai_unsloth_CUDA | `save.py:L100-200` | LoRA-only adapter export |
| unslothai_unsloth_ollama_modelfile | [→](./implementations/unslothai_unsloth_ollama_modelfile.md) | ✅Principle:unslothai_unsloth_Ollama_Export, ✅Env:unslothai_unsloth_CUDA | `ollama_template_mappers.py:L1-2192` | Ollama Modelfile generation |
| unslothai_unsloth_load_and_validate | [→](./implementations/unslothai_unsloth_load_and_validate.md) | ✅Principle:unslothai_unsloth_Export_Validation, ✅Env:unslothai_unsloth_CUDA | Various | Export validation pattern |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
