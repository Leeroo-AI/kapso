# Implementation: boft/model.py

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/boft/model.py`
- **Size**: 131 lines
- **Description**: BOFT model and layer injection logic

## Overview

BOFTModel implements the model-level logic for injecting BOFT layers into target modules, handling layer creation and configuration.

## Core Class: BOFTModel

### Attributes

```python
class BOFTModel(BaseTuner):
    prefix: str = "boft_"
    tuner_layer_cls = BOFTLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_BOFT_TARGET_MODULES_MAPPING
```

### Key Methods

**_create_and_replace**:
1. Prepare kwargs with BOFT parameters
2. Check if target is already BOFTLayer
3. If not, create new module
4. If yes, update existing layer with new adapter

**_create_new_module**:
- Supports `torch.nn.Linear` and `torch.nn.Conv2d`
- Creates `Linear` or `Conv2d` BOFT layer wrapper
- Validates fan_in_fan_out setting

### Design Pattern

**Progressive Adapter Addition**:
```python
if not isinstance(target, BOFTLayer):
    new_module = self._create_new_module(...)  # Wrap
else:
    target.update_layer(...)  # Add adapter to existing
```

**Benefit**: Supports multiple BOFT adapters on same layer

## Cross-References

- **Config**: `boft/config.py`
- **Layer**: `boft/layer.py` (implementation details)
