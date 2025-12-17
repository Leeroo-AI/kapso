# Implementation: constants.py

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/utils/constants.py`
- **Size**: 362 lines
- **Module**: `peft.utils.constants`
- **Description**: Model architecture constants and target module mappings for PEFT adapters

## Overview

This module provides comprehensive mappings and constants that enable PEFT (Parameter-Efficient Fine-Tuning) adapters to automatically identify target modules across different transformer architectures. It serves as the central configuration hub for adapter injection, containing mappings for LoRA, IA3, AdaLoRA, and many other adapter types across 30+ model architectures.

## Key Components

### Prefix Tuning Postprocessing Functions

**bloom_model_postprocess_past_key_value**
- **Purpose**: Transforms past key-value caches for BLOOM models in prefix tuning
- **Process**: Reshapes (layers, batch, heads, tokens, dim) → tuple of (K, V) pairs
- **Special Handling**: Splits total_layers in half for keys/values, transposes and reshapes appropriately

**starcoder_model_postprocess_past_key_value**
- **Purpose**: Handles StarCoder/GPT-BigCode model cache transformations
- **Process**: Permutes dimensions and reshapes to flatten last two dimensions
- **Compatibility**: Only used for transformers < 4.54.0

### Target Module Mappings

**TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING**
- **Coverage**: 40+ model types (T5, BART, GPT-2, BLOOM, LLaMA, Mistral, Gemma, etc.)
- **Purpose**: Defines default attention projection layers for LoRA injection
- **Pattern**: Primarily targets query/value projections (q_proj, v_proj, q, v, query, value)
- **Examples**:
  - LLaMA/Mistral: `["q_proj", "v_proj"]`
  - BERT/RoBERTa: `["query", "value"]`
  - GPT-2: `["c_attn"]`
  - T5: `["q", "v"]`

**Adapter-Specific Mappings** (inherited from LoRA with modifications):
- **BOFT, BONE, C3A, DELORA, HRA, LOHA, LOKR, MISS, OFT, POLY, RANDLORA, ROAD**: Identical to LoRA
- **FOURIERFT, SHIRA, VERA**: Modified versions excluding certain modules or adjusting for specific architectures
- **C3A**: Targets MLP layers for GPT models (`mlp.c_proj`)

**TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING**
- **Purpose**: Defines target modules for (IA)³ adapter (query, key, value + feedforward layers)
- **Pattern**: Includes both attention (k_proj, v_proj) and MLP layers (down_proj, fc2, etc.)
- **Examples**:
  - LLaMA: `["k_proj", "v_proj", "down_proj"]`
  - T5: `["k", "v", "wo"]`
  - Mixtral: `["k_proj", "v_proj", "w2"]`

**TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING**
- **Purpose**: Specifies layer normalization modules for LN-tuning
- **Examples**:
  - LLaMA: `["input_layernorm", "post_attention_layernorm", "norm"]`
  - BART: `["self_attn_layer_norm", "encoder_attn_layer_norm", "final_layer_norm"]`
  - Gemma2: Includes pre/post feedforward layer norms

**TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING**
- **Purpose**: Comprehensive module targeting for adaptive rank allocation
- **Coverage**: Attention (Q, K, V, O) and MLP layers (wi, wo, fc1, fc2)

**TRANSFORMERS_MODELS_TO_OSF_TARGET_MODULES_MAPPING**
- **Purpose**: Full projection coverage for Orthogonal Subspace Factorization
- **Pattern**: All projection layers in both attention and MLP blocks

**TRANSFORMERS_MODELS_TO_WAVEFT_TARGET_MODULES_MAPPING**
- **Purpose**: Similar to LoRA with MLP layer targets for certain architectures

### Miscellaneous Constants

```python
WEIGHTS_NAME = "adapter_model.bin"
SAFETENSORS_WEIGHTS_NAME = "adapter_model.safetensors"
CONFIG_NAME = "adapter_config.json"
EMBEDDING_LAYER_NAMES = ["embed_tokens", "lm_head"]
SEQ_CLS_HEAD_NAMES = ["score", "classifier"]
INCLUDE_LINEAR_LAYERS_SHORTHAND = "all-linear"
TOKENIZER_CONFIG_NAME = "tokenizer_config.json"
DUMMY_TARGET_MODULES = "dummy-target-modules"
DUMMY_MODEL_CONFIG = {"model_type": "custom"}
MIN_TARGET_MODULES_FOR_OPTIMIZATION = 20
```

## Technical Details

### Conditional Mapping Construction

**Transformers Version Compatibility**:
```python
transformers_le_4_53 = packaging.version.parse(transformers.__version__) < packaging.version.parse("4.54.0.dev0")
if transformers_le_4_53:
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING["gpt_bigcode"] = starcoder_model_postprocess_past_key_value
```

**BLOOM Special Handling**:
```python
if hasattr(BloomPreTrainedModel, "_convert_to_standard_cache"):
    # Fixed in transformers PR #31445
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING["bloom"] = bloom_model_postprocess_past_key_value
```

### Design Patterns

1. **Centralized Configuration**: Single source of truth for all adapter target modules
2. **Architecture Abstraction**: Provides uniform interface across diverse model architectures
3. **Naming Convention Normalization**: Handles variations in module naming (q_proj vs query, c_attn vs q_proj)
4. **Backward Compatibility**: Maintains support for older transformer versions
5. **Extensibility**: New models/adapters can be added by extending existing dictionaries

### Module Selection Strategy

**Attention-Focused Adapters** (LoRA, BOFT, etc.):
- Target Q and V projections primarily
- Rationale: Maximum impact with minimal parameters

**Comprehensive Adapters** (AdaLoRA, OSF):
- Target all attention projections + MLP layers
- Rationale: Allow for adaptive rank allocation across entire network

**Layer Normalization Adapters**:
- Target normalization layers only
- Rationale: Efficient distribution shift adaptation

## Integration Points

1. **PeftModel**: Uses mappings to auto-identify target modules when user specifies model type
2. **Config Classes**: Reference these mappings as default values for `target_modules`
3. **Layer Creation**: Tuner classes use these mappings during module replacement
4. **Model Saving/Loading**: File name constants used throughout save/load logic

## Usage Examples

### Automatic Target Module Selection
```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# No need to specify target_modules - uses constants.py mapping
config = LoraConfig(r=16, lora_alpha=32)  # Automatically targets ["q_proj", "v_proj"]
peft_model = get_peft_model(model, config)
```

### Custom Target Modules Override
```python
# Override default mapping
config = LoraConfig(
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Add K and O projections
)
```

## Performance Considerations

- **Lazy Evaluation**: Mappings constructed at import time (minimal overhead)
- **Dictionary Lookups**: O(1) access time for model type → target modules
- **Memory Footprint**: ~10KB total for all mapping dictionaries
- **Target Module Optimization**: When `len(target_modules) > MIN_TARGET_MODULES_FOR_OPTIMIZATION`, suffix optimization is applied

## Maintenance Notes

### Adding New Model Support

1. **Identify module names**: Use `model.named_modules()` to find attention layers
2. **Add to appropriate mappings**: Include in LoRA, IA3, and other relevant dictionaries
3. **Test with multiple adapters**: Ensure compatibility across adapter types
4. **Document naming conventions**: Note any architecture-specific quirks

### Version Compatibility Checks

- Use `packaging.version` for numeric version comparisons
- Use `hasattr()` for API presence checks
- Add conditional imports/mappings with clear comments explaining the version cutoff

## Dependencies

- **Core**: `torch`, `transformers`, `packaging`
- **Model-Specific**: `transformers.BloomPreTrainedModel`
- **PEFT Internal**: Used by all tuner modules (lora, ia3, adalora, etc.)

## Cross-References

- **Related Modules**: `peft.config`, `peft.tuners.*`, `peft.mapping`
- **Used By**: `LoraConfig`, `IA3Config`, `AdaLoraConfig`, `BaseTuner._prepare_adapter_config`
- **Extended By**: Custom model integrations can add to these mappings at runtime
