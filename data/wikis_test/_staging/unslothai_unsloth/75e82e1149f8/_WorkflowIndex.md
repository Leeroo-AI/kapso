# Workflow Index: unslothai_unsloth

> Tracks Workflow pages and their connections to Implementations, Principles, etc.
> **Update IMMEDIATELY** after creating or modifying a Workflow page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| unslothai_unsloth_QLoRA_Finetuning | [→](./workflows/unslothai_unsloth_QLoRA_Finetuning.md) | ✅Principle:unslothai_unsloth_QLoRA_4bit_Quantization, ✅Principle:unslothai_unsloth_Low_Rank_Adaptation, ✅Principle:unslothai_unsloth_Supervised_Fine_Tuning, ✅Principle:unslothai_unsloth_Chat_Template_Formatting, ✅Principle:unslothai_unsloth_Gradient_Checkpointing, ✅Principle:unslothai_unsloth_Sample_Packing, ✅Impl:unslothai_unsloth_FastLanguageModel, ✅Impl:unslothai_unsloth_UnslothTrainer, ✅Env:unslothai_unsloth_GPU_CUDA_Environment, ✅Heuristic:unslothai_unsloth_LoRA_Rank_Selection, ✅Heuristic:unslothai_unsloth_Memory_Management, ✅Heuristic:unslothai_unsloth_Learning_Rate_Guidelines, ✅Heuristic:unslothai_unsloth_Dtype_Selection | Main QLoRA fine-tuning workflow for LLMs using 4-bit quantization |
| unslothai_unsloth_Model_Export_GGUF | [→](./workflows/unslothai_unsloth_Model_Export_GGUF.md) | ✅Principle:unslothai_unsloth_LoRA_Weight_Merging, ✅Principle:unslothai_unsloth_GGUF_Model_Quantization, ✅Principle:unslothai_unsloth_Chat_Template_Formatting, ✅Impl:unslothai_unsloth_unsloth_save_model, ✅Impl:unslothai_unsloth_save_to_gguf, ✅Impl:unslothai_unsloth_OLLAMA_TEMPLATES, ✅Env:unslothai_unsloth_GGUF_Export_Environment, ✅Heuristic:unslothai_unsloth_GGUF_Quantization_Selection | Model export to GGUF format for llama.cpp/Ollama deployment |
| unslothai_unsloth_Vision_Model_Finetuning | [→](./workflows/unslothai_unsloth_Vision_Model_Finetuning.md) | ✅Principle:unslothai_unsloth_Vision_Language_Modeling, ✅Principle:unslothai_unsloth_Low_Rank_Adaptation, ✅Principle:unslothai_unsloth_QLoRA_4bit_Quantization, ✅Principle:unslothai_unsloth_Gradient_Checkpointing, ✅Impl:unslothai_unsloth_FastVisionModel, ✅Impl:unslothai_unsloth_UnslothVisionDataCollator, ✅Env:unslothai_unsloth_GPU_CUDA_Environment | Vision-language model fine-tuning for multimodal tasks |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
