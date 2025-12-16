# Principle Index: unslothai_unsloth

> Tracks Principle pages and their connections to Implementations, Workflows, etc.
> **Update IMMEDIATELY** after creating or modifying a Principle page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| unslothai_unsloth_QLoRA_4bit_Quantization | [→](./principles/unslothai_unsloth_QLoRA_4bit_Quantization.md) | ✅Impl:unslothai_unsloth_FastLanguageModel, ✅Impl:unslothai_unsloth_FastVisionModel, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning | 4-bit NormalFloat quantization for memory-efficient LLM fine-tuning |
| unslothai_unsloth_Low_Rank_Adaptation | [→](./principles/unslothai_unsloth_Low_Rank_Adaptation.md) | ✅Impl:unslothai_unsloth_FastLanguageModel, ✅Impl:unslothai_unsloth_FastVisionModel, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Workflow:unslothai_unsloth_Vision_Model_Finetuning | Parameter-efficient fine-tuning with low-rank decomposition |
| unslothai_unsloth_GGUF_Model_Quantization | [→](./principles/unslothai_unsloth_GGUF_Model_Quantization.md) | ✅Impl:unslothai_unsloth_save_to_gguf, ✅Impl:unslothai_unsloth_OLLAMA_TEMPLATES, ✅Workflow:unslothai_unsloth_Model_Export_GGUF | Post-training quantization for CPU/GPU inference deployment |
| unslothai_unsloth_Supervised_Fine_Tuning | [→](./principles/unslothai_unsloth_Supervised_Fine_Tuning.md) | ✅Impl:unslothai_unsloth_UnslothTrainer, ✅Impl:unslothai_unsloth_FastLanguageModel, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning | Training methodology for instruction-following models |
| unslothai_unsloth_Vision_Language_Modeling | [→](./principles/unslothai_unsloth_Vision_Language_Modeling.md) | ✅Impl:unslothai_unsloth_FastVisionModel, ✅Impl:unslothai_unsloth_UnslothVisionDataCollator, ✅Workflow:unslothai_unsloth_Vision_Model_Finetuning | Multimodal architecture combining vision and language |
| unslothai_unsloth_Chat_Template_Formatting | [→](./principles/unslothai_unsloth_Chat_Template_Formatting.md) | ✅Impl:unslothai_unsloth_OLLAMA_TEMPLATES, ✅Impl:unslothai_unsloth_FastLanguageModel, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Workflow:unslothai_unsloth_Model_Export_GGUF | Conversation formatting for training and inference |
| unslothai_unsloth_LoRA_Weight_Merging | [→](./principles/unslothai_unsloth_LoRA_Weight_Merging.md) | ✅Impl:unslothai_unsloth_unsloth_save_model, ✅Impl:unslothai_unsloth_save_to_gguf, ✅Workflow:unslothai_unsloth_Model_Export_GGUF | Combining adapter weights with base model for deployment |
| unslothai_unsloth_Gradient_Checkpointing | [→](./principles/unslothai_unsloth_Gradient_Checkpointing.md) | ✅Impl:unslothai_unsloth_FastLanguageModel, ✅Impl:unslothai_unsloth_FastVisionModel, ✅Impl:unslothai_unsloth_UnslothTrainer, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning | Memory optimization by recomputing activations |
| unslothai_unsloth_Sample_Packing | [→](./principles/unslothai_unsloth_Sample_Packing.md) | ✅Impl:unslothai_unsloth_UnslothTrainer, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning | Efficient training by concatenating short sequences |
| unslothai_unsloth_Gated_Activation_Functions | [→](./principles/unslothai_unsloth_Gated_Activation_Functions.md) | ✅Impl:unslothai_unsloth_geglu_kernel, ✅Impl:unslothai_unsloth_FastLanguageModel, ✅Heuristic:unslothai_unsloth_Dtype_Selection | GLU family activations (GEGLU, SwiGLU) for transformer MLP layers |
| unslothai_unsloth_FP8_Inference_Quantization | [→](./principles/unslothai_unsloth_FP8_Inference_Quantization.md) | ✅Impl:unslothai_unsloth_FP8_Quantization, ✅Heuristic:unslothai_unsloth_Dtype_Selection | 8-bit floating point quantization for memory-efficient inference |
| unslothai_unsloth_Synthetic_Data_Generation | [→](./principles/unslothai_unsloth_Synthetic_Data_Generation.md) | ✅Impl:unslothai_unsloth_SyntheticDataKit, ✅Heuristic:unslothai_unsloth_Memory_Management | LLM-based training data generation from source documents |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
