# src/peft/tuners/xlora/config.py

## Overview
Configuration class for X-LoRA (Mixture of LoRA Experts), defining all parameters needed to set up and control X-LoRA behavior. This configuration extends `PeftConfig` and provides extensive customization options for the X-LoRA classifier and adapter management.

## Class: XLoraConfig

### Key Configuration Parameters

#### Model Architecture
- **hidden_size** (int): Hidden size of the base model, defaults to 4096 if not specified
- **adapters** (dict[str, str]): Mapping of adapter names to their LoRA checkpoint paths
  - Keys are adapter names, values are paths/IDs to load adapters from
  - These adapters are automatically loaded and used as LoRA experts
  - During loading from pretrained, paths are disregarded in favor of saved adapters

#### X-LoRA Classifier Settings
- **xlora_depth** (int, default=1): Depth of the X-LoRA classifier neural network
- **xlora_size** (int, default=2048): Hidden size of classifier (only relevant if depth > 1)
- **xlora_dropout_p** (float, default=0.2): Dropout probability in classifier (only relevant if depth > 1)

#### Scaling Behavior
- **enable_softmax** (bool, default=True): Apply softmax to X-LoRA classifier outputs
- **enable_softmax_topk** (bool, default=False): Apply softmax only to top-k adapters
  - Mutually exclusive with `enable_softmax`
  - Only valid when `top_k_lora` is set
- **softmax_temperature** (float, default=1.0): Temperature for softmax, lower values yield sharper predictions
- **layerwise_scalings** (bool, default=False):
  - If True: Generate unique scalings for each LoRA adapter layer
  - If False: Broadcast same scalings to all layers

#### Sparsity and Optimization
- **top_k_lora** (int, optional): Sparsely select only top-k LoRA experts instead of dense weighting
- **scaling_pass_value** (float, default=0.0): Value to use for scalings during the scaling pass
- **global_scaling_weight** (float, default=1.0): Global multiplier for all LoRA adapter outputs

#### Training Options
- **use_trainable_adapters** (bool, default=False): Make the adapters trainable during X-LoRA training

### Post-Initialization Validation
The `__post_init__` method performs several validation checks:
1. Sets `peft_type` to `PeftType.XLORA`
2. Warns if `hidden_size` is not provided (defaults to 4096)
3. Warns if `adapters` dictionary is not provided (defaults to empty dict)
4. Warns if `enable_softmax_topk` is enabled but `top_k_lora` is not set
5. Warns if both `enable_softmax_topk` and `enable_softmax` are enabled (worse performance)
6. Validates that `top_k_lora` is at least 1 if provided

## Purpose
This configuration class provides complete control over X-LoRA's mixture-of-experts behavior, including:
- How many adapters to use and where to load them from
- The architecture of the gating classifier
- How adapter contributions are weighted and combined
- Sparsity options for computational efficiency
- Training behavior for adapters

## Usage Pattern
The configuration is typically created before initializing an X-LoRA model and passed to `get_peft_model()` or used directly with `XLoraModel`.
