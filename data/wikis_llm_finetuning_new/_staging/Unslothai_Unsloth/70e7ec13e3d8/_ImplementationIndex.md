# Implementation Index: Unslothai_Unsloth

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| Unslothai_Unsloth_ALLOWED_QUANTS | [→](./implementations/Unslothai_Unsloth_ALLOWED_QUANTS.md) | ✅Principle:Unslothai_Unsloth_Quantization_Selection, ✅Env:Unslothai_Unsloth_Ollama | Quantization options constant |
| Unslothai_Unsloth_Attention_Dispatch | [→](./implementations/Unslothai_Unsloth_Attention_Dispatch.md) | — | Attention backend dispatcher |
| Unslothai_Unsloth_Cohere_Model | [→](./implementations/Unslothai_Unsloth_Cohere_Model.md) | — | Cohere model optimizations |
| Unslothai_Unsloth_convert_to_gguf | [→](./implementations/Unslothai_Unsloth_convert_to_gguf.md) | ✅Principle:Unslothai_Unsloth_GGUF_Conversion, ✅Env:Unslothai_Unsloth_Ollama | GGUF export function |
| Unslothai_Unsloth_create_ollama_modelfile | [→](./implementations/Unslothai_Unsloth_create_ollama_modelfile.md) | ✅Principle:Unslothai_Unsloth_Ollama_Template_Generation, ✅Env:Unslothai_Unsloth_Ollama | Modelfile generation |
| Unslothai_Unsloth_Dataset_Preparation_GRPO_Pattern | [→](./implementations/Unslothai_Unsloth_Dataset_Preparation_GRPO_Pattern.md) | ✅Principle:Unslothai_Unsloth_Dataset_Preparation_GRPO | Pattern doc for GRPO data |
| Unslothai_Unsloth_DeepSeek_Registry | [→](./implementations/Unslothai_Unsloth_DeepSeek_Registry.md) | — | DeepSeek model registration |
| Unslothai_Unsloth_Device_Type | [→](./implementations/Unslothai_Unsloth_Device_Type.md) | — | Hardware device detection |
| Unslothai_Unsloth_Falcon_H1_Model | [→](./implementations/Unslothai_Unsloth_Falcon_H1_Model.md) | — | Falcon H1 hybrid model |
| Unslothai_Unsloth_FastLanguageModel_from_pretrained | [→](./implementations/Unslothai_Unsloth_FastLanguageModel_from_pretrained.md) | ✅Principle:Unslothai_Unsloth_Model_Loading, ✅Env:Unslothai_Unsloth_CUDA_11 | Main model loading API |
| Unslothai_Unsloth_FastLanguageModel_from_pretrained_vllm | [→](./implementations/Unslothai_Unsloth_FastLanguageModel_from_pretrained_vllm.md) | ✅Principle:Unslothai_Unsloth_RL_Model_Loading, ✅Env:Unslothai_Unsloth_VLLM | vLLM-enabled loading |
| Unslothai_Unsloth_FastVisionModel_from_pretrained | [→](./implementations/Unslothai_Unsloth_FastVisionModel_from_pretrained.md) | ✅Principle:Unslothai_Unsloth_Vision_Model_Loading, ✅Env:Unslothai_Unsloth_Vision | Vision model loading API |
| Unslothai_Unsloth_Flex_Attention | [→](./implementations/Unslothai_Unsloth_Flex_Attention.md) | ✅Env:Unslothai_Unsloth_CUDA_11, ✅Heuristic:Unslothai_Unsloth_Triton_Optimization | Flex attention with softcapping |
| Unslothai_Unsloth_FP8_Kernels | [→](./implementations/Unslothai_Unsloth_FP8_Kernels.md) | ✅Env:Unslothai_Unsloth_CUDA_11, ✅Heuristic:Unslothai_Unsloth_Triton_Optimization | FP8 quantization kernels |
| Unslothai_Unsloth_GEGLU_Kernels | [→](./implementations/Unslothai_Unsloth_GEGLU_Kernels.md) | ✅Env:Unslothai_Unsloth_CUDA_11, ✅Heuristic:Unslothai_Unsloth_Triton_Optimization | GEGLU activation kernel |
| Unslothai_Unsloth_GEMM_Autotuning | [→](./implementations/Unslothai_Unsloth_GEMM_Autotuning.md) | — | Grouped GEMM autotuning |
| Unslothai_Unsloth_GEMM_Backward | [→](./implementations/Unslothai_Unsloth_GEMM_Backward.md) | — | Grouped GEMM backward pass |
| Unslothai_Unsloth_GEMM_Forward | [→](./implementations/Unslothai_Unsloth_GEMM_Forward.md) | — | Grouped GEMM forward pass |
| Unslothai_Unsloth_GEMM_Tuning | [→](./implementations/Unslothai_Unsloth_GEMM_Tuning.md) | — | Manual GEMM tuning configs |
| Unslothai_Unsloth_get_chat_template | [→](./implementations/Unslothai_Unsloth_get_chat_template.md) | ✅Principle:Unslothai_Unsloth_Data_Formatting | Chat template utility |
| Unslothai_Unsloth_get_peft_model | [→](./implementations/Unslothai_Unsloth_get_peft_model.md) | ✅Principle:Unslothai_Unsloth_LoRA_Adapter_Injection, ✅Env:Unslothai_Unsloth_PEFT | LoRA injection API |
| Unslothai_Unsloth_get_peft_model_vision | [→](./implementations/Unslothai_Unsloth_get_peft_model_vision.md) | ✅Principle:Unslothai_Unsloth_Vision_LoRA_Configuration, ✅Env:Unslothai_Unsloth_Vision | Vision LoRA injection |
| Unslothai_Unsloth_Granite_Model | [→](./implementations/Unslothai_Unsloth_Granite_Model.md) | — | Granite model optimizations |
| Unslothai_Unsloth_Grouped_GEMM_Interface | [→](./implementations/Unslothai_Unsloth_Grouped_GEMM_Interface.md) | — | Grouped GEMM main API |
| Unslothai_Unsloth_GRPOTrainer_train | [→](./implementations/Unslothai_Unsloth_GRPOTrainer_train.md) | ✅Principle:Unslothai_Unsloth_GRPO_Training, ✅Env:Unslothai_Unsloth_VLLM | GRPO training loop |
| Unslothai_Unsloth_Import_Fixes | [→](./implementations/Unslothai_Unsloth_Import_Fixes.md) | ✅Env:Unslothai_Unsloth_TRL, ✅Env:Unslothai_Unsloth_PEFT | Library compatibility patches |
| Unslothai_Unsloth_Kernel_Utils | [→](./implementations/Unslothai_Unsloth_Kernel_Utils.md) | — | Triton kernel utilities |
| Unslothai_Unsloth_LayerNorm_Kernel | [→](./implementations/Unslothai_Unsloth_LayerNorm_Kernel.md) | ✅Env:Unslothai_Unsloth_CUDA_11, ✅Heuristic:Unslothai_Unsloth_Triton_Optimization | LayerNorm Triton kernel |
| Unslothai_Unsloth_Llama4_MoE_Layer | [→](./implementations/Unslothai_Unsloth_Llama4_MoE_Layer.md) | — | Llama4 MoE reference layer |
| Unslothai_Unsloth_Model_Registry | [→](./implementations/Unslothai_Unsloth_Model_Registry.md) | — | Core model registry |
| Unslothai_Unsloth_MoE_Block | [→](./implementations/Unslothai_Unsloth_MoE_Block.md) | — | MoE block implementation |
| Unslothai_Unsloth_MoE_Ops | [→](./implementations/Unslothai_Unsloth_MoE_Ops.md) | — | MoE common operations |
| Unslothai_Unsloth_Multimodal_Data_Preparation_Pattern | [→](./implementations/Unslothai_Unsloth_Multimodal_Data_Preparation_Pattern.md) | ✅Principle:Unslothai_Unsloth_Multimodal_Data_Preparation | Pattern doc for vision data |
| Unslothai_Unsloth_Qwen3_MoE_Layer | [→](./implementations/Unslothai_Unsloth_Qwen3_MoE_Layer.md) | — | Qwen3 MoE reference layer |
| Unslothai_Unsloth_Qwen3_MoE_Model | [→](./implementations/Unslothai_Unsloth_Qwen3_MoE_Model.md) | — | Qwen3 MoE model support |
| Unslothai_Unsloth_RawTextDataLoader | [→](./implementations/Unslothai_Unsloth_RawTextDataLoader.md) | ✅Env:Unslothai_Unsloth_TRL | Raw text data processing |
| Unslothai_Unsloth_Reward_Function_Interface | [→](./implementations/Unslothai_Unsloth_Reward_Function_Interface.md) | ✅Principle:Unslothai_Unsloth_Reward_Function_Design | Reward function signature |
| Unslothai_Unsloth_RMSNorm_Kernel | [→](./implementations/Unslothai_Unsloth_RMSNorm_Kernel.md) | — | RMSNorm Triton kernel |
| Unslothai_Unsloth_RoPE_Kernel | [→](./implementations/Unslothai_Unsloth_RoPE_Kernel.md) | — | RoPE embedding kernel |
| Unslothai_Unsloth_save_pretrained_merged | [→](./implementations/Unslothai_Unsloth_save_pretrained_merged.md) | ✅Principle:Unslothai_Unsloth_Model_Saving | Model saving API |
| Unslothai_Unsloth_SFTConfig | [→](./implementations/Unslothai_Unsloth_SFTConfig.md) | ✅Principle:Unslothai_Unsloth_Training_Configuration, ✅Env:Unslothai_Unsloth_TRL | Training hyperparameters |
| Unslothai_Unsloth_SFTTrainer_train | [→](./implementations/Unslothai_Unsloth_SFTTrainer_train.md) | ✅Principle:Unslothai_Unsloth_SFT_Training, ✅Env:Unslothai_Unsloth_TRL | SFT training loop |
| Unslothai_Unsloth_SwiGLU_Kernel | [→](./implementations/Unslothai_Unsloth_SwiGLU_Kernel.md) | — | SwiGLU activation kernel |
| Unslothai_Unsloth_SyntheticDataKit | [→](./implementations/Unslothai_Unsloth_SyntheticDataKit.md) | ✅Env:Unslothai_Unsloth_VLLM | Synthetic data generation |
| Unslothai_Unsloth_UnslothVisionDataCollator | [→](./implementations/Unslothai_Unsloth_UnslothVisionDataCollator.md) | ✅Principle:Unslothai_Unsloth_Vision_Training_Setup, ✅Env:Unslothai_Unsloth_Vision | Vision batch collation |
