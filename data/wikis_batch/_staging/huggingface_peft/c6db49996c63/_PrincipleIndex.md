# Principle Index: huggingface_peft

> Tracks Principle pages and their connections to Implementations, Workflows, etc.
> **Update IMMEDIATELY** after creating or modifying a Principle page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| huggingface_peft_Model_Loading | [→](./principles/huggingface_peft_Model_Loading.md) | ✅Impl:huggingface_peft_AutoModel_from_pretrained, ✅Workflow:huggingface_peft_LoRA_Finetuning | Loading pretrained base model |
| huggingface_peft_LoRA_Configuration | [→](./principles/huggingface_peft_LoRA_Configuration.md) | ✅Impl:huggingface_peft_LoraConfig, ✅Workflow:huggingface_peft_LoRA_Finetuning | Configuring LoRA parameters |
| huggingface_peft_PEFT_Application | [→](./principles/huggingface_peft_PEFT_Application.md) | ✅Impl:huggingface_peft_get_peft_model, ✅Workflow:huggingface_peft_LoRA_Finetuning | Applying PEFT to model |
| huggingface_peft_Adapter_Training | [→](./principles/huggingface_peft_Adapter_Training.md) | ✅Impl:huggingface_peft_Training_Loop, ✅Workflow:huggingface_peft_LoRA_Finetuning | Training adapter weights |
| huggingface_peft_Adapter_Saving | [→](./principles/huggingface_peft_Adapter_Saving.md) | ✅Impl:huggingface_peft_save_pretrained, ✅Workflow:huggingface_peft_LoRA_Finetuning | Saving trained adapters |
| huggingface_peft_Adapter_Loading | [→](./principles/huggingface_peft_Adapter_Loading.md) | ✅Impl:huggingface_peft_PeftModel_from_pretrained, ✅Workflow:huggingface_peft_Adapter_Inference | Loading trained adapters |
| huggingface_peft_Adapter_Merging | [→](./principles/huggingface_peft_Adapter_Merging.md) | ✅Impl:huggingface_peft_merge_and_unload, ✅Workflow:huggingface_peft_Adapter_Inference | Merging adapter into base |
| huggingface_peft_Adapter_Addition | [→](./principles/huggingface_peft_Adapter_Addition.md) | ✅Impl:huggingface_peft_load_adapter, ✅Workflow:huggingface_peft_Multi_Adapter_Management | Loading additional adapters |
| huggingface_peft_Adapter_Switching | [→](./principles/huggingface_peft_Adapter_Switching.md) | ✅Impl:huggingface_peft_set_adapter, ✅Workflow:huggingface_peft_Multi_Adapter_Management | Switching active adapter |
| huggingface_peft_Adapter_Combination | [→](./principles/huggingface_peft_Adapter_Combination.md) | ✅Impl:huggingface_peft_add_weighted_adapter, ✅Workflow:huggingface_peft_Multi_Adapter_Management | Combining multiple adapters |
| huggingface_peft_Adapter_Lifecycle | [→](./principles/huggingface_peft_Adapter_Lifecycle.md) | ✅Impl:huggingface_peft_delete_adapter, ✅Workflow:huggingface_peft_Multi_Adapter_Management | Managing adapter lifecycle |
| huggingface_peft_Hotswap_Preparation | [→](./principles/huggingface_peft_Hotswap_Preparation.md) | ✅Impl:huggingface_peft_prepare_model_for_compiled_hotswap, ✅Workflow:huggingface_peft_Adapter_Hotswapping | Preparing for hotswap |
| huggingface_peft_Hotswap_Execution | [→](./principles/huggingface_peft_Hotswap_Execution.md) | ✅Impl:huggingface_peft_hotswap_adapter, ✅Workflow:huggingface_peft_Adapter_Hotswapping | Executing hotswap |
| huggingface_peft_Quantization_Config | [→](./principles/huggingface_peft_Quantization_Config.md) | ✅Impl:huggingface_peft_BitsAndBytesConfig, ✅Workflow:huggingface_peft_QLoRA_Training | Configuring quantization |
| huggingface_peft_Memory_Optimization | [→](./principles/huggingface_peft_Memory_Optimization.md) | ✅Impl:huggingface_peft_prepare_model_for_kbit_training, ✅Workflow:huggingface_peft_QLoRA_Training | Memory optimization for QLoRA |

---

## Principles by Workflow

### LoRA_Finetuning Workflow
1. ✅ Model_Loading
2. ✅ LoRA_Configuration
3. ✅ PEFT_Application
4. ✅ Adapter_Training
5. ✅ Adapter_Saving

### QLoRA_Training Workflow
1. ✅ Quantization_Config
2. ✅ Memory_Optimization

### Adapter_Inference Workflow
1. ✅ Adapter_Loading
2. ✅ Adapter_Merging

### Multi_Adapter_Management Workflow
1. ✅ Adapter_Addition
2. ✅ Adapter_Switching
3. ✅ Adapter_Combination
4. ✅ Adapter_Lifecycle

### Adapter_Hotswapping Workflow
1. ✅ Hotswap_Preparation
2. ✅ Hotswap_Execution

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
