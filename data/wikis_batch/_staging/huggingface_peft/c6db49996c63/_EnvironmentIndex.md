# Environment Index: huggingface_peft

> Tracks Environment pages and which pages require them.
> **Update IMMEDIATELY** after creating or modifying a Environment page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| huggingface_peft_CUDA_Training | [→](./environments/huggingface_peft_CUDA_Training.md) | ✅Impl:huggingface_peft_get_peft_model, ✅Impl:huggingface_peft_LoraConfig, ✅Impl:huggingface_peft_save_pretrained, ✅Impl:huggingface_peft_PeftModel_from_pretrained, ✅Impl:huggingface_peft_merge_and_unload, ✅Impl:huggingface_peft_load_adapter, ✅Impl:huggingface_peft_set_adapter | Base CUDA environment for PEFT training |
| huggingface_peft_Quantized_Training | [→](./environments/huggingface_peft_Quantized_Training.md) | ✅Impl:huggingface_peft_BitsAndBytesConfig, ✅Impl:huggingface_peft_prepare_model_for_kbit_training | QLoRA environment with bitsandbytes support |

---

## Environment Types

| Type | Count | Description |
|------|-------|-------------|
| Training | 2 | GPU/quantization training environments |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
