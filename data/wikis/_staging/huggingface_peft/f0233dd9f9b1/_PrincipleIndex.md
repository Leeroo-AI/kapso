# Principle Index: huggingface_peft

> Tracks Principle pages and their connections to Implementations, Workflows, etc.
> **Update IMMEDIATELY** after creating or modifying a Principle page.

## Pages

| Page | File | Implementation | Workflows | Notes |
|------|------|----------------|-----------|-------|
| huggingface_peft_Base_Model_Loading | [→](./principles/huggingface_peft_Base_Model_Loading.md) | ✅Impl:AutoModelForCausalLM_from_pretrained | ✅LoRA_Fine_Tuning, ✅QLoRA_Training | Load pretrained model as frozen base |
| huggingface_peft_LoRA_Configuration | [→](./principles/huggingface_peft_LoRA_Configuration.md) | ✅Impl:LoraConfig_init | ✅LoRA_Fine_Tuning, ✅QLoRA_Training | Configure rank, alpha, target modules |
| huggingface_peft_PEFT_Model_Creation | [→](./principles/huggingface_peft_PEFT_Model_Creation.md) | ✅Impl:get_peft_model | ✅LoRA_Fine_Tuning, ✅QLoRA_Training | Inject adapter layers into base model |
| huggingface_peft_Training_Preparation | [→](./principles/huggingface_peft_Training_Preparation.md) | ✅Impl:model_train_mode | ✅LoRA_Fine_Tuning, ✅QLoRA_Training | Set training mode, verify gradients |
| huggingface_peft_Training_Execution | [→](./principles/huggingface_peft_Training_Execution.md) | ✅Impl:Trainer_train | ✅LoRA_Fine_Tuning, ✅QLoRA_Training | Execute training loop on adapters |
| huggingface_peft_Adapter_Serialization | [→](./principles/huggingface_peft_Adapter_Serialization.md) | ✅Impl:PeftModel_save_pretrained | ✅LoRA_Fine_Tuning, ✅QLoRA_Training, ✅Adapter_Merging | Save adapter weights efficiently |
| huggingface_peft_Quantization_Configuration | [→](./principles/huggingface_peft_Quantization_Configuration.md) | ✅Impl:BitsAndBytesConfig_4bit | ✅QLoRA_Training | Configure NF4 4-bit quantization |
| huggingface_peft_Kbit_Training_Preparation | [→](./principles/huggingface_peft_Kbit_Training_Preparation.md) | ✅Impl:prepare_model_for_kbit_training | ✅QLoRA_Training | Prepare quantized model for training |
| huggingface_peft_Adapter_Loading | [→](./principles/huggingface_peft_Adapter_Loading.md) | ✅Impl:PeftModel_from_pretrained | ✅Adapter_Loading_Inference, ✅Adapter_Merging | Load trained adapter for inference |
| huggingface_peft_Adapter_Merging_Into_Base | [→](./principles/huggingface_peft_Adapter_Merging_Into_Base.md) | ✅Impl:merge_and_unload | ✅Adapter_Merging | Permanently merge adapter into base |
| huggingface_peft_Multi_Adapter_Loading | [→](./principles/huggingface_peft_Multi_Adapter_Loading.md) | ✅Impl:load_adapter | ✅Multi_Adapter_Management, ✅Adapter_Merging | Load additional adapters |
| huggingface_peft_Adapter_Merge_Execution | [→](./principles/huggingface_peft_Adapter_Merge_Execution.md) | ✅Impl:add_weighted_adapter | ✅Adapter_Merging | TIES/DARE adapter combination |
| huggingface_peft_Adapter_Switching | [→](./principles/huggingface_peft_Adapter_Switching.md) | ✅Impl:set_adapter | ✅Multi_Adapter_Management | Switch active adapter |
| huggingface_peft_Adapter_Enable_Disable | [→](./principles/huggingface_peft_Adapter_Enable_Disable.md) | ✅Impl:disable_adapter_context | ✅Multi_Adapter_Management | Temporarily bypass adapters |
| huggingface_peft_Adapter_Deletion | [→](./principles/huggingface_peft_Adapter_Deletion.md) | ✅Impl:delete_adapter | ✅Multi_Adapter_Management | Remove adapter to free memory |
| huggingface_peft_Adapter_State_Query | [→](./principles/huggingface_peft_Adapter_State_Query.md) | ✅Impl:query_adapter_state | ✅Multi_Adapter_Management | Query active adapter and configs |
| huggingface_peft_Inference_Configuration | [→](./principles/huggingface_peft_Inference_Configuration.md) | ✅Impl:model_eval | ✅Adapter_Loading_Inference | Set eval mode for inference |
| huggingface_peft_Inference_Execution | [→](./principles/huggingface_peft_Inference_Execution.md) | ✅Impl:model_generate | ✅Adapter_Loading_Inference | Execute inference with adapter |
| huggingface_peft_Merge_Evaluation | [→](./principles/huggingface_peft_Merge_Evaluation.md) | ✅Impl:merged_adapter_evaluation | ✅Adapter_Merging | Evaluate merged adapter quality |
| huggingface_peft_Merge_Strategy_Configuration | [→](./principles/huggingface_peft_Merge_Strategy_Configuration.md) | ✅Impl:merge_strategy_selection | ✅Adapter_Merging | Configure TIES/DARE merge params |
| huggingface_peft_QLoRA_Configuration | [→](./principles/huggingface_peft_QLoRA_Configuration.md) | ✅Impl:LoraConfig_for_qlora | ✅QLoRA_Training | LoRA config for quantized training |
| huggingface_peft_QLoRA_Training_Execution | [→](./principles/huggingface_peft_QLoRA_Training_Execution.md) | ✅Impl:Trainer_train_qlora | ✅QLoRA_Training | Train with quantized model |
| huggingface_peft_Quantized_Model_Loading | [→](./principles/huggingface_peft_Quantized_Model_Loading.md) | ✅Impl:AutoModel_from_pretrained_quantized | ✅QLoRA_Training | Load 4-bit/8-bit quantized model |

---

## Summary

- **Total Principle Pages:** 23
- **All Principles have 1:1 Implementation mapping:** ✅ Yes
- **Concept-only Principles (no API):** 0

## Principle Categories

### Training Principles
- Base_Model_Loading
- LoRA_Configuration
- PEFT_Model_Creation
- Training_Preparation
- Training_Execution
- Adapter_Serialization

### QLoRA-Specific Principles
- Quantization_Configuration
- Kbit_Training_Preparation

### Inference & Deployment Principles
- Adapter_Loading
- Adapter_Merging_Into_Base

### Multi-Adapter Principles
- Multi_Adapter_Loading
- Adapter_Merge_Execution
- Adapter_Switching
- Adapter_Enable_Disable
- Adapter_Deletion

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
