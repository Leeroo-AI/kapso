# Implementation Index: unslothai_unsloth

> Tracks Implementation pages and their connections to Principles, Environments, etc.
> **Update IMMEDIATELY** after creating or modifying a Implementation page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| unslothai_unsloth_FastLanguageModel | [→](./implementations/unslothai_unsloth_FastLanguageModel.md) | ✅Principle:unslothai_unsloth_QLoRA_4bit_Quantization, ✅Principle:unslothai_unsloth_Low_Rank_Adaptation, ✅Principle:unslothai_unsloth_Supervised_Fine_Tuning, ✅Principle:unslothai_unsloth_Chat_Template_Formatting, ✅Principle:unslothai_unsloth_Gradient_Checkpointing, ✅Principle:unslothai_unsloth_Gated_Activation_Functions, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Env:unslothai_unsloth_GPU_CUDA_Environment, ✅Heuristic:unslothai_unsloth_LoRA_Rank_Selection, ✅Heuristic:unslothai_unsloth_Memory_Management, ✅Heuristic:unslothai_unsloth_Dtype_Selection | unsloth/models/loader.py:L120-L621 |
| unslothai_unsloth_FastVisionModel | [→](./implementations/unslothai_unsloth_FastVisionModel.md) | ✅Principle:unslothai_unsloth_QLoRA_4bit_Quantization, ✅Principle:unslothai_unsloth_Low_Rank_Adaptation, ✅Principle:unslothai_unsloth_Vision_Language_Modeling, ✅Principle:unslothai_unsloth_Gradient_Checkpointing, ✅Workflow:unslothai_unsloth_Vision_Model_Finetuning, ✅Env:unslothai_unsloth_GPU_CUDA_Environment, ✅Heuristic:unslothai_unsloth_LoRA_Rank_Selection, ✅Heuristic:unslothai_unsloth_Dtype_Selection | unsloth/models/loader.py:L1257-L1258; unsloth/models/vision.py:L316-L800 |
| unslothai_unsloth_unsloth_save_model | [→](./implementations/unslothai_unsloth_unsloth_save_model.md) | ✅Principle:unslothai_unsloth_LoRA_Weight_Merging, ✅Workflow:unslothai_unsloth_Model_Export_GGUF | unsloth/save.py:L228-L851 |
| unslothai_unsloth_save_to_gguf | [→](./implementations/unslothai_unsloth_save_to_gguf.md) | ✅Principle:unslothai_unsloth_GGUF_Model_Quantization, ✅Principle:unslothai_unsloth_LoRA_Weight_Merging, ✅Workflow:unslothai_unsloth_Model_Export_GGUF, ✅Env:unslothai_unsloth_GGUF_Export_Environment, ✅Heuristic:unslothai_unsloth_GGUF_Quantization_Selection | unsloth/save.py:L1000-L1500 |
| unslothai_unsloth_UnslothTrainer | [→](./implementations/unslothai_unsloth_UnslothTrainer.md) | ✅Principle:unslothai_unsloth_Supervised_Fine_Tuning, ✅Principle:unslothai_unsloth_Gradient_Checkpointing, ✅Principle:unslothai_unsloth_Sample_Packing, ✅Workflow:unslothai_unsloth_QLoRA_Finetuning, ✅Env:unslothai_unsloth_GPU_CUDA_Environment, ✅Heuristic:unslothai_unsloth_Memory_Management, ✅Heuristic:unslothai_unsloth_Learning_Rate_Guidelines | unsloth/trainer.py:L181-L198 |
| unslothai_unsloth_UnslothVisionDataCollator | [→](./implementations/unslothai_unsloth_UnslothVisionDataCollator.md) | ✅Principle:unslothai_unsloth_Vision_Language_Modeling, ✅Workflow:unslothai_unsloth_Vision_Model_Finetuning | unsloth_zoo/vision_utils.py; unsloth/trainer.py:L36-L37 |
| unslothai_unsloth_OLLAMA_TEMPLATES | [→](./implementations/unslothai_unsloth_OLLAMA_TEMPLATES.md) | ✅Principle:unslothai_unsloth_GGUF_Model_Quantization, ✅Principle:unslothai_unsloth_Chat_Template_Formatting, ✅Workflow:unslothai_unsloth_Model_Export_GGUF | unsloth/ollama_template_mappers.py:L1-L500 |
| unslothai_unsloth_geglu_kernel | [→](./implementations/unslothai_unsloth_geglu_kernel.md) | ✅Principle:unslothai_unsloth_Gated_Activation_Functions, ✅Env:unslothai_unsloth_GPU_CUDA_Environment, ✅Heuristic:unslothai_unsloth_Dtype_Selection | unsloth/kernels/geglu.py:L63-290 - GEGLU activation kernel with exact and approximate modes |
| unslothai_unsloth_FP8_Quantization | [→](./implementations/unslothai_unsloth_FP8_Quantization.md) | ✅Principle:unslothai_unsloth_FP8_Inference_Quantization, ✅Env:unslothai_unsloth_GPU_CUDA_Environment, ✅Heuristic:unslothai_unsloth_Dtype_Selection | unsloth/kernels/fp8.py:L1-599 - FP8 quantization/dequantization kernels |
| unslothai_unsloth_SyntheticDataKit | [→](./implementations/unslothai_unsloth_SyntheticDataKit.md) | ✅Principle:unslothai_unsloth_Synthetic_Data_Generation, ✅Env:unslothai_unsloth_GPU_CUDA_Environment, ✅Heuristic:unslothai_unsloth_Memory_Management | unsloth/dataprep/synthetic.py:L153-465 - Synthetic data generation toolkit |
| unslothai_unsloth_ModelRegistry | [→](./implementations/unslothai_unsloth_ModelRegistry.md) | ✅Env:unslothai_unsloth_GPU_CUDA_Environment | unsloth/registry/registry.py:L1-191 - Model variant registry infrastructure |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
