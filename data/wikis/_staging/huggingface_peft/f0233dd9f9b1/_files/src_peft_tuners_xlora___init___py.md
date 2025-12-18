# src/peft/tuners/xlora/__init__.py

## Overview
This module initializes the X-LoRA (Mixture of LoRA Experts) tuning method for PEFT. X-LoRA is an advanced technique that combines multiple LoRA adapters with a learned gating mechanism to dynamically select and weight different expert adapters based on input.

## Key Components

### Imports
- `XLoraConfig`: Configuration class for X-LoRA settings
- `XLoraModel`: Main model class that implements X-LoRA functionality

### Registration
The module registers X-LoRA as a PEFT method using `register_peft_method()` with:
- **name**: "xlora"
- **config_cls**: XLoraConfig
- **model_cls**: XLoraModel

## Purpose
X-LoRA enables mixture-of-experts style parameter-efficient fine-tuning where multiple LoRA adapters are loaded and a classifier learns to dynamically weight their contributions based on the input. This allows for more flexible and adaptive model behavior compared to single-adapter approaches.

## Exports
The module exports two main classes:
- `XLoraConfig`: For configuring X-LoRA behavior
- `XLoraModel`: For creating and using X-LoRA models

## Reference
X-LoRA method is described in detail in the paper: https://huggingface.co/papers/2402.07148
