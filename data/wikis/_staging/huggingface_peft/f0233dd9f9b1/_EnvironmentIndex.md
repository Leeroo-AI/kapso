# Environment Index: huggingface_peft

> Tracks Environment pages and which pages require them.
> **Update IMMEDIATELY** after creating or modifying a Environment page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| huggingface_peft_Core_Environment | [→](./environments/huggingface_peft_Core_Environment.md) | ✅Impl:huggingface_peft_LoraConfig_init, ✅Impl:huggingface_peft_get_peft_model, ✅Impl:huggingface_peft_PeftModel_from_pretrained, ✅Impl:huggingface_peft_PeftModel_save_pretrained, ✅Impl:huggingface_peft_merge_and_unload | Core PEFT deps: torch>=1.13, transformers, accelerate>=0.21 |
| huggingface_peft_Quantization_Environment | [→](./environments/huggingface_peft_Quantization_Environment.md) | ✅Impl:huggingface_peft_BitsAndBytesConfig_4bit, ✅Impl:huggingface_peft_prepare_model_for_kbit_training | bitsandbytes for 4-bit/8-bit QLoRA |
| huggingface_peft_GPTQ_Environment | [→](./environments/huggingface_peft_GPTQ_Environment.md) | ✅Impl:huggingface_peft_get_peft_model | auto_gptq>=0.5 or gptqmodel>=2.0+optimum>=1.24 |
| huggingface_peft_LoftQ_Environment | [→](./environments/huggingface_peft_LoftQ_Environment.md) | ✅Impl:huggingface_peft_LoraConfig_init | scipy for LoftQ initialization |

---

## Summary

- **Total Environment Pages:** 4
- **Core Environments:** 1 (base PEFT)
- **Quantization Environments:** 3 (bitsandbytes, GPTQ, LoftQ)

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
