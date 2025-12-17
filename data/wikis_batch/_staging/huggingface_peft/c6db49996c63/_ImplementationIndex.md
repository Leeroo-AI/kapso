# Implementation Index: huggingface_peft

> Tracks Implementation pages and their connections to Principles, Environments, etc.
> **Update IMMEDIATELY** after creating or modifying a Implementation page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| huggingface_peft_LoraConfig | [→](./implementations/huggingface_peft_LoraConfig.md) | ✅Principle:huggingface_peft_LoRA_Configuration, ✅Env:huggingface_peft_CUDA_Training, ✅Heuristic:huggingface_peft_DoRA_Mixed_Batch_Limitation | API Doc - LoRA configuration class |
| huggingface_peft_get_peft_model | [→](./implementations/huggingface_peft_get_peft_model.md) | ✅Principle:huggingface_peft_PEFT_Application, ✅Env:huggingface_peft_CUDA_Training | API Doc - Create PEFT model |
| huggingface_peft_save_pretrained | [→](./implementations/huggingface_peft_save_pretrained.md) | ✅Principle:huggingface_peft_Adapter_Saving, ✅Env:huggingface_peft_CUDA_Training | API Doc - Save adapter weights |
| huggingface_peft_AutoModel_from_pretrained | [→](./implementations/huggingface_peft_AutoModel_from_pretrained.md) | ✅Principle:huggingface_peft_Model_Loading, ✅Env:huggingface_peft_CUDA_Training | Wrapper Doc - Load base model |
| huggingface_peft_Training_Loop | [→](./implementations/huggingface_peft_Training_Loop.md) | ✅Principle:huggingface_peft_Adapter_Training, ✅Env:huggingface_peft_CUDA_Training | Wrapper Doc - Training process |
| huggingface_peft_PeftModel_from_pretrained | [→](./implementations/huggingface_peft_PeftModel_from_pretrained.md) | ✅Principle:huggingface_peft_Adapter_Loading, ✅Env:huggingface_peft_CUDA_Training | API Doc - Load adapter |
| huggingface_peft_merge_and_unload | [→](./implementations/huggingface_peft_merge_and_unload.md) | ✅Principle:huggingface_peft_Adapter_Merging, ✅Env:huggingface_peft_CUDA_Training, ✅Heuristic:huggingface_peft_Quantized_Merge_Rounding, ✅Heuristic:huggingface_peft_Safe_Merge_NaN_Check | API Doc - Merge adapter into base |
| huggingface_peft_load_adapter | [→](./implementations/huggingface_peft_load_adapter.md) | ✅Principle:huggingface_peft_Adapter_Addition, ✅Env:huggingface_peft_CUDA_Training | API Doc - Load additional adapters |
| huggingface_peft_set_adapter | [→](./implementations/huggingface_peft_set_adapter.md) | ✅Principle:huggingface_peft_Adapter_Switching, ✅Env:huggingface_peft_CUDA_Training | API Doc - Switch active adapter |
| huggingface_peft_add_weighted_adapter | [→](./implementations/huggingface_peft_add_weighted_adapter.md) | ✅Principle:huggingface_peft_Adapter_Combination, ✅Env:huggingface_peft_CUDA_Training | API Doc - Combine adapters |
| huggingface_peft_delete_adapter | [→](./implementations/huggingface_peft_delete_adapter.md) | ✅Principle:huggingface_peft_Adapter_Lifecycle, ✅Env:huggingface_peft_CUDA_Training | API Doc - Remove adapter |
| huggingface_peft_prepare_model_for_compiled_hotswap | [→](./implementations/huggingface_peft_prepare_model_for_compiled_hotswap.md) | ✅Principle:huggingface_peft_Hotswap_Preparation, ✅Env:huggingface_peft_CUDA_Training | API Doc - Prepare for hotswap |
| huggingface_peft_hotswap_adapter | [→](./implementations/huggingface_peft_hotswap_adapter.md) | ✅Principle:huggingface_peft_Hotswap_Execution, ✅Env:huggingface_peft_CUDA_Training | API Doc - Execute hotswap |
| huggingface_peft_BitsAndBytesConfig | [→](./implementations/huggingface_peft_BitsAndBytesConfig.md) | ✅Principle:huggingface_peft_Quantization_Config, ✅Env:huggingface_peft_Quantized_Training, ✅Heuristic:huggingface_peft_4bit_Defensive_Clone | Wrapper Doc - Quantization config |
| huggingface_peft_prepare_model_for_kbit_training | [→](./implementations/huggingface_peft_prepare_model_for_kbit_training.md) | ✅Principle:huggingface_peft_Memory_Optimization, ✅Env:huggingface_peft_Quantized_Training, ✅Heuristic:huggingface_peft_Gradient_Checkpointing | API Doc - Prepare quantized model |

## Orphan Page Implementations (Phase 6c)

### AdaLoRA Tuner
| Page | File | Source | Notes |
|------|------|--------|-------|
| huggingface_peft_AdaLoraLayer | [→](./implementations/huggingface_peft_AdaLoraLayer.md) | adalora/layer.py | SVD-based adaptive layers |
| huggingface_peft_AdaLoraModel | [→](./implementations/huggingface_peft_AdaLoraModel.md) | adalora/model.py | AdaLoRA model orchestration |
| huggingface_peft_BNB_AdaLoraLinear | [→](./implementations/huggingface_peft_BNB_AdaLoraLinear.md) | adalora/bnb.py | Quantized AdaLoRA |
| huggingface_peft_GPTQ_AdaLoraLinear | [→](./implementations/huggingface_peft_GPTQ_AdaLoraLinear.md) | adalora/gptq.py | GPTQ AdaLoRA |
| adalora_config.py | [→](./implementations/adalora_config.py.md) | adalora/config.py | AdaLoRA config |

### Adaption Prompt Tuner
| Page | File | Source | Notes |
|------|------|--------|-------|
| adaption_prompt_config.py | [→](./implementations/adaption_prompt_config.py.md) | adaption_prompt/config.py | Config |
| adaption_prompt_layer.py | [→](./implementations/adaption_prompt_layer.py.md) | adaption_prompt/layer.py | Layer |
| adaption_prompt_model.py | [→](./implementations/adaption_prompt_model.py.md) | adaption_prompt/model.py | Model |

### BOFT Tuner
| Page | File | Source | Notes |
|------|------|--------|-------|
| huggingface_peft_BOFTLayer | [→](./implementations/huggingface_peft_BOFTLayer.md) | boft/layer.py | Butterfly base layer |
| huggingface_peft_BOFTLinear | [→](./implementations/huggingface_peft_BOFTLinear.md) | boft/layer.py | BOFT for Linear |
| huggingface_peft_BOFTConv2d | [→](./implementations/huggingface_peft_BOFTConv2d.md) | boft/layer.py | BOFT for Conv2d |
| huggingface_peft_FastBlockDiag | [→](./implementations/huggingface_peft_FastBlockDiag.md) | boft/layer.py | CUDA block diagonal |
| boft_config.py | [→](./implementations/boft_config.py.md) | boft/config.py | BOFT config |
| boft_model.py | [→](./implementations/boft_model.py.md) | boft/model.py | BOFT model |

### BONE Tuner (Transitioning to MISS)
| Page | File | Source | Notes |
|------|------|--------|-------|
| bone_tuner_implementation | [→](./implementations/bone_tuner_implementation.md) | bone/*.py | BONE layer/config/model (deprecated, use MISS) |

### C3A Tuner
| Page | File | Source | Notes |
|------|------|--------|-------|
| c3a_config | [→](./implementations/c3a_config.md) | c3a/config.py | C3A config |
| c3a_layer | [→](./implementations/c3a_layer.md) | c3a/layer.py | C3A layer |
| c3a_model | [→](./implementations/c3a_model.md) | c3a/model.py | C3A model |

### CPT Tuner
| Page | File | Source | Notes |
|------|------|--------|-------|
| cpt_config.py | [→](./implementations/cpt_config.py.md) | cpt/config.py | CPT config |

### GraLoRA Tuner
| Page | File | Source | Notes |
|------|------|--------|-------|
| gralora_tuner_implementation | [→](./implementations/gralora_tuner_implementation.md) | gralora/*.py | GraLoRA layer/config/model |

### HRA Tuner
| Page | File | Source | Notes |
|------|------|--------|-------|
| hra_tuner_implementation | [→](./implementations/hra_tuner_implementation.md) | hra/*.py | HRA layer/config/model |

### IA3 Tuner
| Page | File | Source | Notes |
|------|------|--------|-------|
| ia3_tuner_implementation | [→](./implementations/ia3_tuner_implementation.md) | ia3/*.py | IA3 layer/config/model |

### LoHa Tuner
| Page | File | Source | Notes |
|------|------|--------|-------|
| loha_tuner_implementation | [→](./implementations/loha_tuner_implementation.md) | loha/*.py | LoHa layer/config/model |

### LyCORIS Base
| Page | File | Source | Notes |
|------|------|--------|-------|
| lycoris_utils.py | [→](./implementations/lycoris_utils.py.md) | lycoris_utils.py | LyCORIS base classes |

### Multitask Prompt Tuning
| Page | File | Source | Notes |
|------|------|--------|-------|
| multitask_prompt_tuning_config.py | [→](./implementations/multitask_prompt_tuning_config.py.md) | multitask_prompt_tuning/config.py | Config |
| multitask_prompt_tuning_model.py | [→](./implementations/multitask_prompt_tuning_model.py.md) | multitask_prompt_tuning/model.py | Model |

### OFT Tuner
| Page | File | Source | Notes |
|------|------|--------|-------|
| huggingface_peft_OFTLayer | [→](./implementations/huggingface_peft_OFTLayer.md) | oft/layer.py | OFT base layer |
| huggingface_peft_OFTLinear | [→](./implementations/huggingface_peft_OFTLinear.md) | oft/layer.py | OFT for Linear |
| huggingface_peft_OFTConv2d | [→](./implementations/huggingface_peft_OFTConv2d.md) | oft/layer.py | OFT for Conv2d |
| huggingface_peft_OFTRotationModule | [→](./implementations/huggingface_peft_OFTRotationModule.md) | oft/layer.py | Rotation computation |
| huggingface_peft_OFTLinear8bitLt | [→](./implementations/huggingface_peft_OFTLinear8bitLt.md) | oft/bnb.py | 8-bit OFT |
| huggingface_peft_OFTLinear4bit | [→](./implementations/huggingface_peft_OFTLinear4bit.md) | oft/bnb.py | 4-bit OFT |
| huggingface_peft_AQLM_OFTLinear | [→](./implementations/huggingface_peft_AQLM_OFTLinear.md) | oft/aqlm.py | AQLM OFT |
| huggingface_peft_AWQ_OFTLinear | [→](./implementations/huggingface_peft_AWQ_OFTLinear.md) | oft/awq.py | AWQ OFT |
| huggingface_peft_EETQ_OFTLinear | [→](./implementations/huggingface_peft_EETQ_OFTLinear.md) | oft/eetq.py | EETQ OFT |
| huggingface_peft_GPTQ_OFTLinear | [→](./implementations/huggingface_peft_GPTQ_OFTLinear.md) | oft/gptq.py | GPTQ OFT |
| huggingface_peft_HQQ_OFTLinear | [→](./implementations/huggingface_peft_HQQ_OFTLinear.md) | oft/hqq.py | HQQ OFT |
| huggingface_peft_INC_OFTLinear | [→](./implementations/huggingface_peft_INC_OFTLinear.md) | oft/inc.py | Intel NC OFT |
| oft_config.py | [→](./implementations/oft_config.py.md) | oft/config.py | OFT config |
| oft_model.py | [→](./implementations/oft_model.py.md) | oft/model.py | OFT model |

### Poly Tuner
| Page | File | Source | Notes |
|------|------|--------|-------|
| poly_config | [→](./implementations/poly_config.md) | poly/config.py | Poly config |
| poly_layer | [→](./implementations/poly_layer.md) | poly/layer.py | Poly layer |
| poly_model | [→](./implementations/poly_model.md) | poly/model.py | Poly model |

### SHiRA Tuner
| Page | File | Source | Notes |
|------|------|--------|-------|
| shira_config | [→](./implementations/shira_config.md) | shira/config.py | SHiRA config |
| shira_layer | [→](./implementations/shira_layer.md) | shira/layer.py | SHiRA layer |
| shira_model | [→](./implementations/shira_model.md) | shira/model.py | SHiRA model |

### VBLoRA Tuner
| Page | File | Source | Notes |
|------|------|--------|-------|
| vblora_config | [→](./implementations/vblora_config.md) | vblora/config.py | VBLoRA config |
| vblora_layer | [→](./implementations/vblora_layer.md) | vblora/layer.py | VBLoRA layer |
| vblora_model | [→](./implementations/vblora_model.md) | vblora/model.py | VBLoRA model |

### VeRA Tuner
| Page | File | Source | Notes |
|------|------|--------|-------|
| vera_config | [→](./implementations/vera_config.md) | vera/config.py | VeRA config |
| vera_layer | [→](./implementations/vera_layer.md) | vera/layer.py | VeRA layer |
| vera_model | [→](./implementations/vera_model.md) | vera/model.py | VeRA model |

### LoRA Variants
| Page | File | Source | Notes |
|------|------|--------|-------|
| huggingface_peft_LoraVariants | [→](./implementations/huggingface_peft_LoraVariants.md) | lora/variants.py | All variant classes |
| huggingface_peft_DoraLayers | [→](./implementations/huggingface_peft_DoraLayers.md) | lora/dora.py | DoRA layers |
| huggingface_peft_ArrowLinearVariant | [→](./implementations/huggingface_peft_ArrowLinearVariant.md) | lora/arrow.py | MoE routing |
| huggingface_peft_EVA | [→](./implementations/huggingface_peft_EVA.md) | lora/eva.py | EVA initialization |
| huggingface_peft_CorDA | [→](./implementations/huggingface_peft_CorDA.md) | lora/corda.py | CorDA initialization |
| huggingface_peft_LoraParallelLinear | [→](./implementations/huggingface_peft_LoraParallelLinear.md) | lora/tp_layer.py | Tensor parallel |

### LoRA Quantization Adapters
| Page | File | Source | Notes |
|------|------|--------|-------|
| huggingface_peft_AQLM_LoraLinear | [→](./implementations/huggingface_peft_AQLM_LoraLinear.md) | lora/aqlm.py | AQLM LoRA |
| huggingface_peft_AWQ_LoraLinear | [→](./implementations/huggingface_peft_AWQ_LoraLinear.md) | lora/awq.py | AWQ LoRA |
| huggingface_peft_EETQ_LoraLinear | [→](./implementations/huggingface_peft_EETQ_LoraLinear.md) | lora/eetq.py | EETQ LoRA |
| huggingface_peft_GPTQ_LoraLinear | [→](./implementations/huggingface_peft_GPTQ_LoraLinear.md) | lora/gptq.py | GPTQ LoRA |
| huggingface_peft_HQQ_LoraLinear | [→](./implementations/huggingface_peft_HQQ_LoraLinear.md) | lora/hqq.py | HQQ LoRA |
| huggingface_peft_INC_LoraLinear | [→](./implementations/huggingface_peft_INC_LoraLinear.md) | lora/inc.py | Intel NC LoRA |
| huggingface_peft_TorchAO_LoraLinear | [→](./implementations/huggingface_peft_TorchAO_LoraLinear.md) | lora/torchao.py | TorchAO LoRA |

### Other Tuners
| Page | File | Source | Notes |
|------|------|--------|-------|
| huggingface_peft_BNB_IA3Linear | [→](./implementations/huggingface_peft_BNB_IA3Linear.md) | ia3/bnb.py | Quantized IA3 |
| huggingface_peft_BNB_RandLoraLinear | [→](./implementations/huggingface_peft_BNB_RandLoraLinear.md) | randlora/bnb.py | Quantized RandLoRA |
| huggingface_peft_BNB_RoadLinear | [→](./implementations/huggingface_peft_BNB_RoadLinear.md) | road/bnb.py | Quantized RoAd |
| huggingface_peft_BNB_VeRALinear | [→](./implementations/huggingface_peft_BNB_VeRALinear.md) | vera/bnb.py | Quantized VeRA |
| huggingface_peft_RandLoraLayer | [→](./implementations/huggingface_peft_RandLoraLayer.md) | randlora/layer.py | RandLoRA layer |
| huggingface_peft_RandLoraModel | [→](./implementations/huggingface_peft_RandLoraModel.md) | randlora/model.py | RandLoRA model |
| huggingface_peft_RandLoraConfig | [→](./implementations/huggingface_peft_RandLoraConfig.md) | randlora/config.py | RandLoRA config |
| huggingface_peft_RoadLayer | [→](./implementations/huggingface_peft_RoadLayer.md) | road/layer.py | RoAd layer |
| huggingface_peft_RoadConfig | [→](./implementations/huggingface_peft_RoadConfig.md) | road/config.py | RoAd config |
| huggingface_peft_RoadModel | [→](./implementations/huggingface_peft_RoadModel.md) | road/model.py | RoAd model |
| huggingface_peft_MissLayer | [→](./implementations/huggingface_peft_MissLayer.md) | miss/layer.py | MISS layer |
| huggingface_peft_MissConfig | [→](./implementations/huggingface_peft_MissConfig.md) | miss/config.py | MISS config |
| huggingface_peft_MissModel | [→](./implementations/huggingface_peft_MissModel.md) | miss/model.py | MISS model |
| huggingface_peft_FourierFTLayer | [→](./implementations/huggingface_peft_FourierFTLayer.md) | fourierft/layer.py | FourierFT layer |
| huggingface_peft_FourierFTConfig | [→](./implementations/huggingface_peft_FourierFTConfig.md) | fourierft/config.py | FourierFT config |
| huggingface_peft_FourierFTModel | [→](./implementations/huggingface_peft_FourierFTModel.md) | fourierft/model.py | FourierFT model |
| huggingface_peft_MethodComparisonApp | [→](./implementations/huggingface_peft_MethodComparisonApp.md) | method_comparison/app.py | Visualization app |
| huggingface_peft_MultiplicativeDropoutLayer | [→](./implementations/huggingface_peft_MultiplicativeDropoutLayer.md) | oft/layer.py | Dropout for OFT |

### Utilities
| Page | File | Source | Notes |
|------|------|--------|-------|
| constants.py | [→](./implementations/constants.py.md) | utils/constants.py | Model constants |
| incremental_pca.py | [→](./implementations/incremental_pca.py.md) | utils/incremental_pca.py | Incremental PCA |
| loftq_utils.py | [→](./implementations/loftq_utils.py.md) | utils/loftq_utils.py | LoftQ utilities |
| peft_types.py | [→](./implementations/peft_types.py.md) | utils/peft_types.py | Type enums |
| functional.py | [→](./implementations/functional.py.md) | functional.py | Functional API |
| helpers.py | [→](./implementations/helpers.py.md) | helpers.py | Helper functions |
| lorafa.py | [→](./implementations/lorafa.py.md) | optimizers/lorafa.py | LoRA-FA optimizer |
| loraplus.py | [→](./implementations/loraplus.py.md) | optimizers/loraplus.py | LoRA+ scheduler |

---

## Implementation Types

| Type | Count | Description |
|------|-------|-------------|
| API Doc | 100+ | Direct PEFT library APIs |
| Wrapper Doc | 3 | External library wrappers (transformers) |
| Pattern Doc | 0 | Common implementation patterns |
| External Tool Doc | 1 | Method comparison visualization |
| **Total Pages** | **106** | All implementation pages |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
