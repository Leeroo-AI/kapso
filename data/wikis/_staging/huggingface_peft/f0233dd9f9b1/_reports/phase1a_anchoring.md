# Phase 1a: Anchoring Report

## Summary
- Workflows created: 5
- Total steps documented: 31

## Workflows Created

| Workflow | Source Files | Steps | Rough APIs |
|----------|--------------|-------|------------|
| huggingface_peft_LoRA_Fine_Tuning | peft_model.py, mapping_func.py, tuners/lora/* | 6 | AutoModel, LoraConfig, get_peft_model, save_pretrained |
| huggingface_peft_QLoRA_Training | helpers.py, tuners/lora/bnb.py, loftq_utils.py | 7 | BitsAndBytesConfig, prepare_model_for_kbit_training, get_peft_model |
| huggingface_peft_Adapter_Loading_Inference | peft_model.py, auto.py, save_and_load.py | 5 | PeftModel.from_pretrained, merge_and_unload, generate |
| huggingface_peft_Adapter_Merging | merge_utils.py, tuners/lora/model.py, peft_model.py | 7 | load_adapter, add_weighted_adapter, TIES, DARE |
| huggingface_peft_Multi_Adapter_Management | peft_model.py, hotswap.py, mixed_model.py | 6 | load_adapter, set_adapter, disable_adapter, delete_adapter |

## Coverage Summary
- Source files with workflow coverage: 45+
- Core files covered: peft_model.py, mapping_func.py, tuners/lora/*, utils/*
- External dependencies identified: transformers, bitsandbytes

## Source Files Identified Per Workflow

### huggingface_peft_LoRA_Fine_Tuning
- `src/peft/tuners/lora/config.py` - LoraConfig dataclass with all adapter parameters
- `src/peft/tuners/lora/model.py` - LoraModel class with adapter injection
- `src/peft/tuners/lora/layer.py` - Core Linear, Conv2d LoRA layer implementations
- `src/peft/mapping_func.py` - get_peft_model() factory function
- `src/peft/peft_model.py` - PeftModel wrapper, save_pretrained, training methods
- `src/peft/utils/save_and_load.py` - get_peft_model_state_dict, serialization

### huggingface_peft_QLoRA_Training
- `src/peft/helpers.py` - prepare_model_for_kbit_training()
- `src/peft/tuners/lora/bnb.py` - LoRA layers for 4-bit/8-bit quantized models
- `src/peft/tuners/lora/gptq.py` - LoRA GPTQ integration
- `src/peft/tuners/lora/hqq.py` - LoRA HQQ integration
- `src/peft/tuners/lora/awq.py` - LoRA AWQ integration
- `src/peft/tuners/lora/eetq.py` - LoRA EETQ integration
- `src/peft/utils/integrations.py` - External library compatibility helpers
- `src/peft/utils/loftq_utils.py` - LoftQ quantization initialization

### huggingface_peft_Adapter_Loading_Inference
- `src/peft/peft_model.py` - PeftModel.from_pretrained(), merge_and_unload()
- `src/peft/auto.py` - AutoPeftModel, AutoPeftModelForCausalLM
- `src/peft/utils/save_and_load.py` - load_peft_weights()
- `src/peft/utils/hotswap.py` - Runtime adapter hot-swapping

### huggingface_peft_Adapter_Merging
- `src/peft/utils/merge_utils.py` - ties(), dare_ties(), dare_linear(), task_arithmetic()
- `src/peft/tuners/lora/model.py` - add_weighted_adapter() method
- `src/peft/peft_model.py` - load_adapter(), save_pretrained()

### huggingface_peft_Multi_Adapter_Management
- `src/peft/peft_model.py` - set_adapter(), disable_adapter(), delete_adapter()
- `src/peft/tuners/tuners_utils.py` - BaseTuner adapter management
- `src/peft/utils/hotswap.py` - Hot-swapping adapters at runtime
- `src/peft/mixed_model.py` - PeftMixedModel for heterogeneous adapter types

## Unique Principles Identified

1. **Base_Model_Loading** - Loading pretrained transformer models
2. **LoRA_Configuration** - Setting up LoraConfig with rank, alpha, target_modules
3. **PEFT_Model_Creation** - Wrapping models with get_peft_model()
4. **Training_Preparation** - Setting up optimizers, enabling gradients
5. **Training_Execution** - Running training loops
6. **Adapter_Serialization** - Saving adapter weights
7. **Quantization_Configuration** - Setting up BitsAndBytesConfig
8. **Quantized_Model_Loading** - Loading models in 4-bit/8-bit
9. **Kbit_Training_Preparation** - prepare_model_for_kbit_training()
10. **QLoRA_Configuration** - LoRA config for quantized training
11. **QLoRA_Training_Execution** - Training with gradient accumulation
12. **Adapter_Loading** - Loading saved adapters
13. **Inference_Configuration** - Setting up eval mode
14. **Inference_Execution** - Running generation/forward passes
15. **Adapter_Merging_Into_Base** - merge_and_unload()
16. **Multi_Adapter_Loading** - Loading multiple adapters
17. **Merge_Strategy_Configuration** - Selecting TIES/DARE/linear
18. **Adapter_Merge_Execution** - add_weighted_adapter()
19. **Merge_Evaluation** - Testing merged adapter performance
20. **Adapter_Switching** - set_adapter() for runtime switching
21. **Adapter_Enable_Disable** - disable_adapter() context manager
22. **Adapter_Deletion** - delete_adapter() for cleanup
23. **Adapter_State_Query** - Inspecting adapter state

## Notes for Phase 1b (Enrichment)

### Files Needing Line-by-Line Tracing
- `src/peft/peft_model.py` - Critical file with many methods (from_pretrained, load_adapter, merge_and_unload, save_pretrained)
- `src/peft/tuners/lora/model.py` - LoraModel.add_weighted_adapter() for merging
- `src/peft/utils/merge_utils.py` - TIES/DARE algorithm implementations
- `src/peft/mapping_func.py` - get_peft_model() factory logic

### External APIs to Document (Wrapper Docs)
- `transformers.AutoModelForCausalLM.from_pretrained()` - Base model loading
- `transformers.BitsAndBytesConfig` - Quantization configuration
- `transformers.Trainer` - Training loop integration

### Key Classes to Document
- `LoraConfig` - 879 lines with many configuration options
- `PeftModel` - 3387 lines, central class
- `LoraModel` - 872 lines, adapter injection
- `Linear` (LoRA layer) - 2304 lines in layer.py

### Unclear Mappings to Resolve
- Training_Execution: PEFT doesn't provide training loop - relies on external Trainer
- Merge_Evaluation: External evaluation, no PEFT-specific API
- Base_Model_Loading: External transformers API, document PEFT-specific considerations

## Artifacts Created

### Workflow Pages
1. `/workflows/huggingface_peft_LoRA_Fine_Tuning.md`
2. `/workflows/huggingface_peft_QLoRA_Training.md`
3. `/workflows/huggingface_peft_Adapter_Loading_Inference.md`
4. `/workflows/huggingface_peft_Adapter_Merging.md`
5. `/workflows/huggingface_peft_Multi_Adapter_Management.md`

### Updated Index Files
- `_WorkflowIndex.md` - Updated with all 5 workflows
- `_RepoMap_huggingface_peft.md` - Updated Coverage column for 45+ files

## Phase 1a Completion Checklist
- [x] Read Phase 0 report
- [x] Read Repository Map index
- [x] Scan README and key files
- [x] Identify 5 candidate workflows (Golden Paths)
- [x] Create 5 Workflow pages with proper structure
- [x] Update Coverage column in Repository Map
- [x] Write rough WorkflowIndex with steps overview
- [x] Write execution report
