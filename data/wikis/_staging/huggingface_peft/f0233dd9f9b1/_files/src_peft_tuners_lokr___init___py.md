# src/peft/tuners/lokr/__init__.py

## Overview
This module initializes the LoKr (Low-Rank Kronecker Product) tuning method for PEFT. LoKr is a parameter-efficient fine-tuning technique that uses Kronecker product factorization to achieve efficient low-rank adaptation of neural network weights.

## Key Components

### Imports
- `LoKrConfig`: Configuration class for LoKr settings
- `LoKrModel`: Main model class implementing LoKr functionality
- `LoKrLayer`: Base layer class for LoKr adapters
- `Linear`, `Conv2d`: Specific LoKr implementations for different layer types

### Registration
The module registers LoKr as a PEFT method using `register_peft_method()` with:
- **name**: "lokr"
- **config_cls**: LoKrConfig
- **model_cls**: LoKrModel
- **is_mixed_compatible**: True (can be mixed with other PEFT methods)

## Exports
The module exports five main classes:
- `Conv2d`: LoKr implementation for 2D convolutional layers
- `Linear`: LoKr implementation for linear (fully connected) layers
- `LoKrConfig`: Configuration for LoKr behavior
- `LoKrLayer`: Base class for LoKr layers
- `LoKrModel`: Model class for applying LoKr

## Purpose
LoKr enables parameter-efficient fine-tuning by decomposing weight updates using Kronecker product factorization. This approach is particularly effective for large weight matrices as it can achieve similar expressiveness to full-rank updates while using significantly fewer parameters.

## Mixed Adapter Compatibility
Unlike some PEFT methods, LoKr is marked as mixed-compatible, meaning it can be used in combination with other PEFT methods on the same model.

## References
- Original method: https://huggingface.co/papers/2108.06098
- Extended implementation: https://huggingface.co/papers/2309.14859
- Based on LyCORIS implementation: https://github.com/KohakuBlueleaf/LyCORIS
