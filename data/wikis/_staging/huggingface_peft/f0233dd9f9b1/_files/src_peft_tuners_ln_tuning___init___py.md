# src/peft/tuners/ln_tuning/__init__.py

## Overview
This module initializes the LN Tuning (LayerNorm Tuning) method for PEFT. LN Tuning is an extremely parameter-efficient fine-tuning technique that only trains the LayerNorm (or other normalization) layers while keeping the rest of the model frozen.

## Key Components

### Imports
- `LNTuningConfig`: Configuration class for LN Tuning settings
- `LNTuningModel`: Main model class implementing LN Tuning functionality

### Registration
The module registers LN Tuning as a PEFT method using `register_peft_method()` with:
- **name**: "ln_tuning"
- **config_cls**: LNTuningConfig
- **model_cls**: LNTuningModel

## Purpose
LN Tuning enables parameter-efficient fine-tuning by only making normalization layers trainable. This approach:
- Requires minimal parameters (typically <1% of model parameters)
- Allows for quick adaptation to new tasks
- Maintains most of the model's pre-trained knowledge
- Works well for scenarios where full fine-tuning is too expensive

## Exports
The module exports two main classes:
- `LNTuningConfig`: Configuration for LN Tuning behavior
- `LNTuningModel`: Model class for applying LN Tuning

## Method Background
LayerNorm tuning is based on the observation that normalization layers play a critical role in model behavior and can effectively adapt models to new domains/tasks with minimal parameter updates.

## Reference
The method is described in detail in: https://huggingface.co/papers/2312.11420
