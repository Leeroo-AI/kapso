# Implementation Index: Unslothai_Unsloth

> Tracks Implementation pages and their connections to Principles, Environments, etc.
> **Update IMMEDIATELY** after creating or modifying an Implementation page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| Unslothai_Unsloth_FastLanguageModel_from_pretrained | [→](./implementations/Unslothai_Unsloth_FastLanguageModel_from_pretrained.md) | ✅Principle:Unslothai_Unsloth_Model_Loading, ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | API Doc - Model loading |
| Unslothai_Unsloth_get_peft_model | [→](./implementations/Unslothai_Unsloth_get_peft_model.md) | ✅Principle:Unslothai_Unsloth_LoRA_Configuration, ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | API Doc - LoRA setup |
| Unslothai_Unsloth_get_chat_template | [→](./implementations/Unslothai_Unsloth_get_chat_template.md) | ✅Principle:Unslothai_Unsloth_Data_Formatting, ✅Principle:Unslothai_Unsloth_Chat_Template_Configuration, ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | API Doc - Chat templates |
| Unslothai_Unsloth_UnslothTrainingArguments | [→](./implementations/Unslothai_Unsloth_UnslothTrainingArguments.md) | ✅Principle:Unslothai_Unsloth_Training_Configuration, ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | Wrapper Doc - Training config |
| Unslothai_Unsloth_SFTTrainer_train | [→](./implementations/Unslothai_Unsloth_SFTTrainer_train.md) | ✅Principle:Unslothai_Unsloth_Supervised_Finetuning, ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | Wrapper Doc - SFT training |
| Unslothai_Unsloth_save_pretrained | [→](./implementations/Unslothai_Unsloth_save_pretrained.md) | ✅Principle:Unslothai_Unsloth_Model_Saving, ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | API Doc - Model saving |
| Unslothai_Unsloth_FastLanguageModel_from_pretrained_vllm | [→](./implementations/Unslothai_Unsloth_FastLanguageModel_from_pretrained_vllm.md) | ✅Principle:Unslothai_Unsloth_RL_Model_Loading, ✅Env:Unslothai_Unsloth_CUDA_GPU_vLLM_Environment | API Doc - vLLM-enabled loading |
| Unslothai_Unsloth_get_peft_model_rl | [→](./implementations/Unslothai_Unsloth_get_peft_model_rl.md) | ✅Principle:Unslothai_Unsloth_RL_LoRA_Configuration, ✅Env:Unslothai_Unsloth_CUDA_GPU_vLLM_Environment | API Doc - RL LoRA config |
| Unslothai_Unsloth_dataset_mapping_pattern | [→](./implementations/Unslothai_Unsloth_dataset_mapping_pattern.md) | ✅Principle:Unslothai_Unsloth_RL_Dataset_Preparation, ✅Env:Unslothai_Unsloth_CUDA_GPU_vLLM_Environment | Pattern Doc - Dataset mapping |
| Unslothai_Unsloth_reward_function_pattern | [→](./implementations/Unslothai_Unsloth_reward_function_pattern.md) | ✅Principle:Unslothai_Unsloth_Reward_Functions, ✅Env:Unslothai_Unsloth_CUDA_GPU_vLLM_Environment | Pattern Doc - Reward functions |
| Unslothai_Unsloth_train_on_responses_only | [→](./implementations/Unslothai_Unsloth_train_on_responses_only.md) | ✅Principle:Unslothai_Unsloth_SFT_Pretraining, ✅Env:Unslothai_Unsloth_CUDA_GPU_vLLM_Environment | API Doc - Response masking |
| Unslothai_Unsloth_UnslothGRPOConfig | [→](./implementations/Unslothai_Unsloth_UnslothGRPOConfig.md) | ✅Principle:Unslothai_Unsloth_GRPO_Configuration, ✅Env:Unslothai_Unsloth_CUDA_GPU_vLLM_Environment | Wrapper Doc - GRPO config |
| Unslothai_Unsloth_UnslothGRPOTrainer | [→](./implementations/Unslothai_Unsloth_UnslothGRPOTrainer.md) | ✅Principle:Unslothai_Unsloth_GRPO_Execution, ✅Env:Unslothai_Unsloth_CUDA_GPU_vLLM_Environment | Wrapper Doc - GRPO training |
| Unslothai_Unsloth_FastVisionModel_from_pretrained | [→](./implementations/Unslothai_Unsloth_FastVisionModel_from_pretrained.md) | ✅Principle:Unslothai_Unsloth_Vision_Model_Loading, ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | API Doc - VLM loading |
| Unslothai_Unsloth_FastBaseModel_get_peft_model | [→](./implementations/Unslothai_Unsloth_FastBaseModel_get_peft_model.md) | ✅Principle:Unslothai_Unsloth_Vision_LoRA_Configuration, ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | API Doc - VLM LoRA |
| Unslothai_Unsloth_multimodal_dataset_pattern | [→](./implementations/Unslothai_Unsloth_multimodal_dataset_pattern.md) | ✅Principle:Unslothai_Unsloth_Multimodal_Data_Preparation, ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | Pattern Doc - Multimodal data |
| Unslothai_Unsloth_FastBaseModel_for_training | [→](./implementations/Unslothai_Unsloth_FastBaseModel_for_training.md) | ✅Principle:Unslothai_Unsloth_Vision_Training_Mode, ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | API Doc - Training mode |
| Unslothai_Unsloth_SFTTrainer_vision | [→](./implementations/Unslothai_Unsloth_SFTTrainer_vision.md) | ✅Principle:Unslothai_Unsloth_Vision_Training, ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | Wrapper Doc - VLM training |
| Unslothai_Unsloth_FastBaseModel_for_inference | [→](./implementations/Unslothai_Unsloth_FastBaseModel_for_inference.md) | ✅Principle:Unslothai_Unsloth_Vision_Inference_Mode, ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | API Doc - Inference mode |
| Unslothai_Unsloth_save_pretrained_vision | [→](./implementations/Unslothai_Unsloth_save_pretrained_vision.md) | ✅Principle:Unslothai_Unsloth_Vision_Model_Saving, ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | API Doc - VLM saving |
| Unslothai_Unsloth_unsloth_save_model_merged | [→](./implementations/Unslothai_Unsloth_unsloth_save_model_merged.md) | ✅Principle:Unslothai_Unsloth_Model_Preparation, ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | API Doc - Model merging |
| Unslothai_Unsloth_ALLOWED_QUANTS | [→](./implementations/Unslothai_Unsloth_ALLOWED_QUANTS.md) | ✅Principle:Unslothai_Unsloth_Quantization_Selection, ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | API Doc - Quant methods |
| Unslothai_Unsloth_save_to_gguf | [→](./implementations/Unslothai_Unsloth_save_to_gguf.md) | ✅Principle:Unslothai_Unsloth_GGUF_Export, ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | API Doc - GGUF conversion |
| Unslothai_Unsloth_OLLAMA_TEMPLATES | [→](./implementations/Unslothai_Unsloth_OLLAMA_TEMPLATES.md) | ✅Principle:Unslothai_Unsloth_Ollama_Modelfile, ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | Pattern Doc - Modelfiles |
| Unslothai_Unsloth_push_to_hub_gguf | [→](./implementations/Unslothai_Unsloth_push_to_hub_gguf.md) | ✅Principle:Unslothai_Unsloth_GGUF_Hub_Upload, ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | API Doc - Hub upload |
| Unslothai_Unsloth_llama_cli_validation | [→](./implementations/Unslothai_Unsloth_llama_cli_validation.md) | ✅Principle:Unslothai_Unsloth_GGUF_Verification, ✅Env:Unslothai_Unsloth_llama_cpp_Environment | External Tool Doc - Validation |
| Unslothai_Unsloth_CLI | [→](./implementations/Unslothai_Unsloth_CLI.md) | ✅Principle:Unslothai_Unsloth_CLI_Data_Loading, ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | CLI entry point |
| Unslothai_Unsloth_RawTextDataLoader | [→](./implementations/Unslothai_Unsloth_RawTextDataLoader.md) | ✅Principle:Unslothai_Unsloth_CLI_Data_Loading, ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | Raw text dataset loading |
| Unslothai_Unsloth_SyntheticDataKit | [→](./implementations/Unslothai_Unsloth_SyntheticDataKit.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_vLLM_Environment | Synthetic Q&A generation |
| Unslothai_Unsloth_Import_Fixes | [→](./implementations/Unslothai_Unsloth_Import_Fixes.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | Compatibility patches |
| Unslothai_Unsloth_Flex_Attention | [→](./implementations/Unslothai_Unsloth_Flex_Attention.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | Logit softcapping attention |
| Unslothai_Unsloth_FP8_Kernels | [→](./implementations/Unslothai_Unsloth_FP8_Kernels.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | FP8 quantization kernels |
| Unslothai_Unsloth_GEGLU_Kernels | [→](./implementations/Unslothai_Unsloth_GEGLU_Kernels.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | GEGLU activation kernels |
| Unslothai_Unsloth_LayerNorm_Kernel | [→](./implementations/Unslothai_Unsloth_LayerNorm_Kernel.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | Fast LayerNorm |
| Unslothai_Unsloth_SwiGLU_Kernels | [→](./implementations/Unslothai_Unsloth_SwiGLU_Kernels.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | SwiGLU activation kernels |
| Unslothai_Unsloth_Kernel_Utils | [→](./implementations/Unslothai_Unsloth_Kernel_Utils.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | Kernel utilities |
| Unslothai_Unsloth_Grouped_GEMM_Interface | [→](./implementations/Unslothai_Unsloth_Grouped_GEMM_Interface.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | MoE GEMM interface |
| Unslothai_Unsloth_Grouped_GEMM_Autotuning | [→](./implementations/Unslothai_Unsloth_Grouped_GEMM_Autotuning.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | MoE autotuning configs |
| Unslothai_Unsloth_Grouped_GEMM_Backward | [→](./implementations/Unslothai_Unsloth_Grouped_GEMM_Backward.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | MoE backward kernels |
| Unslothai_Unsloth_Grouped_GEMM_Forward | [→](./implementations/Unslothai_Unsloth_Grouped_GEMM_Forward.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | MoE forward kernels |
| Unslothai_Unsloth_Grouped_GEMM_Tuning | [→](./implementations/Unslothai_Unsloth_Grouped_GEMM_Tuning.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | Manual tuning configs |
| Unslothai_Unsloth_Llama4_MoE_Layer | [→](./implementations/Unslothai_Unsloth_Llama4_MoE_Layer.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | LLaMA 4 MoE reference |
| Unslothai_Unsloth_Qwen3_MoE_Layer | [→](./implementations/Unslothai_Unsloth_Qwen3_MoE_Layer.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | Qwen 3 MoE reference |
| Unslothai_Unsloth_MoE_Block | [→](./implementations/Unslothai_Unsloth_MoE_Block.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | Generic MoE block |
| Unslothai_Unsloth_MoE_Ops | [→](./implementations/Unslothai_Unsloth_MoE_Ops.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | MoE routing ops |
| Unslothai_Unsloth_FastCohereModel | [→](./implementations/Unslothai_Unsloth_FastCohereModel.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | Cohere model patches |
| Unslothai_Unsloth_FastFalconH1Model | [→](./implementations/Unslothai_Unsloth_FastFalconH1Model.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | Falcon H1 model patches |
| Unslothai_Unsloth_FastGemmaModel | [→](./implementations/Unslothai_Unsloth_FastGemmaModel.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | Gemma model patches |
| Unslothai_Unsloth_FastGemma2Model | [→](./implementations/Unslothai_Unsloth_FastGemma2Model.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | Gemma 2 model patches |
| Unslothai_Unsloth_FastGraniteModel | [→](./implementations/Unslothai_Unsloth_FastGraniteModel.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | Granite model patches |
| Unslothai_Unsloth_FastMistralModel | [→](./implementations/Unslothai_Unsloth_FastMistralModel.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | Mistral model patches |
| Unslothai_Unsloth_FastQwen2Model | [→](./implementations/Unslothai_Unsloth_FastQwen2Model.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | Qwen 2 model patches |
| Unslothai_Unsloth_FastQwen3Model | [→](./implementations/Unslothai_Unsloth_FastQwen3Model.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | Qwen 3 model patches |
| Unslothai_Unsloth_FastQwen3MoeModel | [→](./implementations/Unslothai_Unsloth_FastQwen3MoeModel.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | Qwen 3 MoE patches |
| Unslothai_Unsloth_Device_Type | [→](./implementations/Unslothai_Unsloth_Device_Type.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | GPU device detection |
| Unslothai_Unsloth_Model_Registry | [→](./implementations/Unslothai_Unsloth_Model_Registry.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | Model registry system |
| Unslothai_Unsloth_Attention_Dispatch | [→](./implementations/Unslothai_Unsloth_Attention_Dispatch.md) | ✅Env:Unslothai_Unsloth_CUDA_GPU_Environment | Attention backend dispatch |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
