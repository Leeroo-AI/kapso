# Implementation: oft/model.py

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/oft/model.py`
- **Size**: 199 lines
- **Description**: OFT model with multi-backend support

## Overview

OFTModel implements orthogonal finetuning with support for multiple quantization backends (GPTQ, AWQ, AQLM, HQQ, INC, EETQ, BNB 4/8-bit) and standard layers.

## Core Class: OFTModel

### Attributes

```python
class OFTModel(BaseTuner):
    prefix: str = "oft_"
    tuner_layer_cls = OFTLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_OFT_TARGET_MODULES_MAPPING
```

### Multi-Backend Dispatcher

**_create_new_module**:
```python
dispatchers = [
    dispatch_bnb_8bit,      # bitsandbytes 8-bit
    dispatch_bnb_4bit,      # bitsandbytes 4-bit NF4
    dispatch_eetq,          # EETQ quantization
    dispatch_aqlm,          # AQLM quantization
    dispatch_awq,           # AWQ quantization
    dispatch_gptq,          # GPTQ quantization
    dispatch_hqq,           # Half-Quadratic Quantization
    dispatch_inc,           # Intel Neural Compressor
    dispatch_default,       # Standard nn.Linear/Conv2d
]

for dispatcher in dispatchers:
    new_module = dispatcher(target, adapter_name, oft_config=oft_config, **kwargs)
    if new_module is not None:  # First match wins
        break
```

**Design**: Chain of Responsibility pattern for backend selection

### Merge Restrictions

**_check_merge_allowed**:
```python
def _check_merge_allowed(self):
    super()._check_merge_allowed()
    if getattr(self.model, "quantization_method", None) == "gptq":
        raise ValueError("Cannot merge OFT layers when model is GPTQ quantized")
    if self.peft_config.get("layer_replication"):
        raise ValueError("Cannot merge OFT layers when base model layers are replicated")
```

**Restrictions**:
- GPTQ: Merging not supported
- Layer replication: Merging would duplicate merged weights

## Cross-References

- **Config**: `oft/config.py`
- **Dispatchers**: `oft/gptq.py`, `oft/awq.py`, `oft/bnb.py`, etc.
