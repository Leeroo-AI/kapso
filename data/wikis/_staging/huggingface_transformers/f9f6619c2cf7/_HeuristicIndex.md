# Heuristic Index: huggingface_transformers

> Tracks Heuristic pages and which pages they apply to.
> **Update IMMEDIATELY** after creating or modifying a Heuristic page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| huggingface_transformers_Gradient_Checkpointing | [→](./heuristics/huggingface_transformers_Gradient_Checkpointing.md) | ✅Impl:huggingface_transformers_Training_execution, ✅Impl:huggingface_transformers_TrainingArguments_setup, ✅Workflow:huggingface_transformers_Model_Training_Trainer | Reduce VRAM 50-60% at cost of 20% slower training |
| huggingface_transformers_Batch_Size_Optimization | [→](./heuristics/huggingface_transformers_Batch_Size_Optimization.md) | ✅Impl:huggingface_transformers_TrainingArguments_setup, ✅Impl:huggingface_transformers_Training_execution, ✅Workflow:huggingface_transformers_Model_Training_Trainer | Use gradient accumulation for effective large batches |
| huggingface_transformers_Mixed_Precision_Selection | [→](./heuristics/huggingface_transformers_Mixed_Precision_Selection.md) | ✅Impl:huggingface_transformers_TrainingArguments_setup, ✅Impl:huggingface_transformers_Training_execution, ✅Workflow:huggingface_transformers_Model_Training_Trainer | bf16 for Ampere+ GPUs, fp16 for older architectures |
| huggingface_transformers_Quantization_Selection | [→](./heuristics/huggingface_transformers_Quantization_Selection.md) | ✅Impl:huggingface_transformers_BitsAndBytesConfig_setup, ✅Impl:huggingface_transformers_AutoHfQuantizer_dispatch, ✅Workflow:huggingface_transformers_Model_Quantization | 4-bit NF4 for max savings, 8-bit for better quality |
| huggingface_transformers_Device_Map_Strategy | [→](./heuristics/huggingface_transformers_Device_Map_Strategy.md) | ✅Impl:huggingface_transformers_Accelerate_dispatch, ✅Impl:huggingface_transformers_Model_initialization, ✅Workflow:huggingface_transformers_Model_Loading | device_map="auto" for intelligent GPU distribution |
| huggingface_transformers_Fast_Tokenizer_Usage | [→](./heuristics/huggingface_transformers_Fast_Tokenizer_Usage.md) | ✅Impl:huggingface_transformers_PreTrainedTokenizerBase_from_pretrained, ✅Impl:huggingface_transformers_Tokenizer_encode, ✅Impl:huggingface_transformers_Batch_padding | use_fast=True for 10-100x faster tokenization |
| huggingface_transformers_Liger_Kernel_Optimization | [→](./heuristics/huggingface_transformers_Liger_Kernel_Optimization.md) | ✅Impl:huggingface_transformers_Training_execution, ✅Impl:huggingface_transformers_TrainingArguments_setup, ✅Workflow:huggingface_transformers_Model_Training_Trainer | 20% faster training, 60% memory reduction with fused kernels |
| huggingface_transformers_Safetensors_Preference | [→](./heuristics/huggingface_transformers_Safetensors_Preference.md) | ✅Impl:huggingface_transformers_Weight_loading, ✅Impl:huggingface_transformers_Model_saving, ✅Workflow:huggingface_transformers_Model_Loading | 2-4x faster loading, secure format over pickle |
| huggingface_transformers_Warning_Deprecated_ModelCard | [→](./heuristics/huggingface_transformers_Warning_Deprecated_ModelCard.md) | ✅Impl:huggingface_transformers_ModelCard | ⚠️ ModelCard class deprecated, will be removed in v5 |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
