# Phase 1: Anchoring Report

## Summary

- **Workflows created:** 5
- **Total steps documented:** 29
- **Implementation hints captured:** 16
- **Source files covered:** 18

## Workflows Created

| Workflow | Source Files | Steps | Implementation APIs |
|----------|--------------|-------|---------------------|
| huggingface_peft_LoRA_Finetuning | peft_model.py, mapping_func.py, lora/config.py | 5 | get_peft_model, LoraConfig, save_pretrained |
| huggingface_peft_QLoRA_Training | peft_model.py, lora/bnb.py, utils/integrations.py | 7 | BitsAndBytesConfig, get_peft_model, prepare_model_for_kbit_training |
| huggingface_peft_Adapter_Inference | peft_model.py, auto.py, save_and_load.py | 5 | PeftModel.from_pretrained, merge_and_unload |
| huggingface_peft_Multi_Adapter_Management | peft_model.py, lora/model.py, merge_utils.py | 6 | load_adapter, set_adapter, add_weighted_adapter |
| huggingface_peft_Adapter_Hotswapping | utils/hotswap.py | 6 | hotswap_adapter, prepare_model_for_compiled_hotswap |

## Coverage Summary

- **Core PEFT files covered:** 18 source files
- **Key modules documented:**
  - `src/peft/peft_model.py` - Central to all workflows
  - `src/peft/mapping_func.py` - get_peft_model factory
  - `src/peft/tuners/lora/` - LoRA implementation
  - `src/peft/utils/hotswap.py` - Hot-swapping utilities
  - `src/peft/utils/save_and_load.py` - Serialization

## Implementation Context Captured

| Workflow | Principles | API Docs | Wrapper Docs | Pattern Docs |
|----------|------------|----------|--------------|--------------|
| LoRA_Finetuning | 5 | 3 | 2 | 0 |
| QLoRA_Training | 7 | 4 | 3 | 0 |
| Adapter_Inference | 5 | 4 | 1 | 0 |
| Multi_Adapter_Management | 6 | 5 | 1 | 0 |
| Adapter_Hotswapping | 6 | 4 | 1 | 1 |

## Notes for Excavation Phase

### APIs to Extract (with Source Locations)

| API | Source | Used By Principles |
|-----|--------|-------------------|
| `get_peft_model` | `src/peft/mapping_func.py:L30-128` | PEFT_Application |
| `LoraConfig` | `src/peft/tuners/lora/config.py:L122-879` | LoRA_Configuration, QLoRA_Configuration |
| `PeftModel.from_pretrained` | `src/peft/peft_model.py:L389-700` | Adapter_Loading |
| `PeftModel.save_pretrained` | `src/peft/peft_model.py:L190-387` | Adapter_Saving |
| `model.merge_and_unload` | `src/peft/peft_model.py` | Adapter_Merging |
| `model.load_adapter` | `src/peft/peft_model.py` | Adapter_Addition |
| `model.set_adapter` | `src/peft/peft_model.py` | Adapter_Switching |
| `model.add_weighted_adapter` | `src/peft/tuners/lora/model.py` | Adapter_Combination |
| `model.delete_adapter` | `src/peft/peft_model.py` | Adapter_Lifecycle |
| `hotswap_adapter` | `src/peft/utils/hotswap.py` | Hotswap_Execution |
| `prepare_model_for_compiled_hotswap` | `src/peft/utils/hotswap.py:L56-80` | Hotswap_Preparation |
| `_get_padded_linear` | `src/peft/utils/hotswap.py:L83-147` | Rank_Padding |
| `get_peft_model_state_dict` | `src/peft/utils/save_and_load.py:L57-85` | Adapter_Saving |
| `set_peft_model_state_dict` | `src/peft/utils/save_and_load.py` | Adapter_Loading |
| `BaseTuner` | `src/peft/tuners/tuners_utils.py` | PEFT_Application |
| `LoraLayer` | `src/peft/tuners/lora/layer.py` | LoRA_Configuration |

### External Dependencies to Document

| Library | APIs Used | Workflows |
|---------|-----------|-----------|
| `transformers` | `AutoModelForCausalLM.from_pretrained`, `Trainer`, `BitsAndBytesConfig` | All workflows |
| `bitsandbytes` | `Linear4bit`, quantization backend | QLoRA_Training |
| `accelerate` | Distributed training utilities | LoRA_Finetuning, QLoRA_Training |
| `safetensors` | `safe_save_file`, `safe_load_file` | All save/load operations |
| `huggingface_hub` | `hf_hub_download`, model upload | Adapter_Inference, Multi_Adapter |

### User-Defined Patterns to Document

| Pattern | Description | Workflow |
|---------|-------------|----------|
| Custom target_modules | Specifying which layers to adapt | LoRA_Finetuning |
| Training loop integration | Custom training vs Trainer | LoRA_Finetuning, QLoRA_Training |
| Validation callback | Monitoring adapter swap success | Adapter_Hotswapping |

## Unique Principles Identified

The following Principles were identified across all workflows (deduplicated):

1. **huggingface_peft_Model_Loading** - Loading base transformer models
2. **huggingface_peft_LoRA_Configuration** - Configuring LoRA adapters
3. **huggingface_peft_PEFT_Application** - Applying PEFT to models
4. **huggingface_peft_Adapter_Training** - Training adapter weights
5. **huggingface_peft_Adapter_Saving** - Saving adapter checkpoints
6. **huggingface_peft_Quantization_Config** - Configuring quantization
7. **huggingface_peft_Quantized_Model_Loading** - Loading quantized models
8. **huggingface_peft_QLoRA_Configuration** - QLoRA-specific config
9. **huggingface_peft_Memory_Optimization** - Memory optimization techniques
10. **huggingface_peft_Adapter_Loading** - Loading adapter checkpoints
11. **huggingface_peft_Inference_Configuration** - Inference mode setup
12. **huggingface_peft_Model_Inference** - Running inference
13. **huggingface_peft_Adapter_Merging** - Merging adapters into base
14. **huggingface_peft_Adapter_Addition** - Adding adapters to model
15. **huggingface_peft_Adapter_Switching** - Switching active adapters
16. **huggingface_peft_Adapter_Combination** - Combining multiple adapters
17. **huggingface_peft_Adapter_Lifecycle** - Managing adapter lifecycle
18. **huggingface_peft_Hotswap_Preparation** - Preparing for hot-swap
19. **huggingface_peft_Torch_Compile_Setup** - torch.compile integration
20. **huggingface_peft_Hotswap_Execution** - Executing hot-swaps
21. **huggingface_peft_Rank_Padding** - Handling rank mismatches
22. **huggingface_peft_Hotswap_Validation** - Validating hot-swaps

## Architecture Insights

### Core Design Patterns

1. **Config → Model → Layer Pattern**: All tuners follow this consistent structure:
   - Configuration dataclass defines parameters
   - Model class handles adapter injection and orchestration
   - Layer class implements the actual forward pass

2. **Wrapper/Delegate Pattern**: PeftModel wraps the base model and delegates to tuner-specific implementations

3. **Plugin Architecture**: Easy to add new adapter types through PEFT_TYPE_TO_TUNER_MAPPING

4. **Multi-Backend Quantization**: Supports bitsandbytes, GPTQ, AWQ, AQLM, EETQ, HQQ, etc.

### Key Entry Points for Phase 2

| Entry Point | Type | Priority |
|-------------|------|----------|
| `get_peft_model()` | Factory function | Critical |
| `PeftModel.from_pretrained()` | Class method | Critical |
| `LoraConfig` | Configuration | High |
| `hotswap_adapter()` | Utility function | High |
| `merge_and_unload()` | Instance method | Medium |

## Recommendations for Phase 2

1. **Start with Core APIs**: Focus on `get_peft_model`, `PeftModel.from_pretrained`, and `LoraConfig` as they're used in all LoRA-based workflows

2. **Document Quantization Integration**: The bitsandbytes integration in `lora/bnb.py` is complex and critical for QLoRA users

3. **Trace Hot-Swap Implementation**: The hot-swapping code has nuanced handling of rank padding and torch.compile compatibility

4. **Consider External Dependencies**: Many operations (model loading, training) use transformers APIs - document PEFT's wrappers around these

## Files Not Covered

The following important files were not directly covered by workflows:
- Other tuner implementations (BOFT, IA3, VeRA, etc.)
- Optimizer implementations (LoRA-FA, LoRA+)
- Test files (comprehensive but not user-facing)

These could be candidates for additional workflows or advanced documentation.
