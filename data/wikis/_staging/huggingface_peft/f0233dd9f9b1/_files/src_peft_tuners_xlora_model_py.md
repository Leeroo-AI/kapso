# src/peft/tuners/xlora/model.py

## Overview
Implementation of X-LoRA (Mixture of LoRA Experts), a parameter-efficient fine-tuning method that combines multiple LoRA adapters with a learned classifier to dynamically weight their contributions. This file contains the main `XLoraModel` class and supporting functions.

## Key Components

### Function: convert_layers_to_xlora
Converts standard LoRA layers to X-LoRA layers by wrapping them with X-LoRA functionality.

**Process:**
1. Iterates through all modules in the base model
2. Identifies LoRA layers (Linear, Embedding, Conv2d)
3. Wraps each with corresponding X-LoRA layer (XLoraLinearLayer, XLoraEmbeddingLayer, XLoraConv2dLayer)
4. Replaces the forward method to intercept calls
5. Tracks device and total number of swapped layers

**Returns:** Tuple of (total_swapped_layers, device)

### Function: _load_adapter_into_lora_model
Loads a LoRA adapter into the X-LoRA model's internal LoRA model structure.

**Key Operations:**
1. Loads adapter configuration from pretrained checkpoint
2. Sets inference mode to False (adapters need to be callable)
3. Injects adapter into the model
4. Loads adapter weights and remaps keys to proper structure
5. Validates no unexpected keys exist
6. Casts adapter dtype if needed

### Class: XLoraModel

Inherits from `BaseTuner` and implements the X-LoRA mixture-of-experts architecture.

#### Initialization
**Parameters:**
- **model**: Base PyTorch model to apply X-LoRA to
- **config**: XLoraConfig object with all settings
- **adapter_name**: Name for the X-LoRA adapter
- **torch_device**: Device to load adapters on
- **ephemeral_gpu_offload**: Whether to use ephemeral GPU offloading
- **autocast_adapter_dtype**: Whether to autocast adapter weights

**Initialization Process:**
1. Creates internal LoraModel with dummy target modules
2. Loads all LoRA expert adapters from config
3. Freezes adapters if `use_trainable_adapters` is False
4. Converts LoRA layers to X-LoRA layers
5. Creates X-LoRA classifier for gating
6. Initializes internal state (scalings, disabled flag)

**Validation:**
- Raises error if `use_cache=True` (X-LoRA requires hidden states)

#### Core Methods

**_enable_peft_forward_hooks()**
Context manager that implements X-LoRA's two-pass forward system:

**Pass 1 - Scaling Pass:**
1. Disables adapter layers temporarily
2. Runs forward pass with dummy scalings to get hidden states
3. Feeds hidden states to X-LoRA classifier to compute real scalings
4. Stores computed scalings for retrieval

**Pass 2 - Real Forward:**
1. Re-enables adapter layers
2. Injects computed scalings into all LoRA layers via forward hooks
3. Runs actual forward pass with scaled adapters

**generate()**
Wrapper for generation that:
- Forces `use_cache=False`
- Calls internal lora_model's generate method
- Re-freezes adapters after generation

#### Scaling Control Methods

**set_topk_lora(value)**: Set sparse top-k selection for adapters
**set_global_scaling_weight(weight)**: Set global multiplier for all adapter outputs
**set_scaling_pass_value(value)**: Set dummy scaling value for first pass
**get_global_scaling_weight()**: Retrieve global scaling weight
**get_latest_scalings()**: Get most recent scaling predictions (batch_size, seq_len, n_layers, n_classes)

#### Logging Methods

**enable_scalings_logging()**: Start recording scaling predictions
**disable_scalings_logging()**: Stop recording (keeps existing log)
**clear_scalings_log()**: Clear all recorded scalings
**get_scalings_log()**: Retrieve list of all recorded scalings
**get_bucketed_scalings_log()**: Get scalings grouped by sequence length

#### Adapter Management

**enable_adapter_layers()**: Enable X-LoRA adapter (sets disabled=False)
**disable_adapter_layers()**: Disable X-LoRA adapter (sets disabled=True)

#### Overridden Methods

**forward()**: Delegates to internal lora_model
**_mark_only_adapters_as_trainable()**: Does nothing (X-LoRA controls this)
**_create_and_replace()**: Does nothing (X-LoRA has no target modules)
**_check_target_module_exists()**: Returns False (X-LoRA has no target modules)

## Architecture Details

### Two-Pass Forward System
X-LoRA uses a unique two-pass architecture:

1. **Dummy Forward Pass**: Run model with minimal adapter scaling to extract hidden states
2. **Classifier Pass**: Feed hidden states to X-LoRA classifier to predict optimal scalings
3. **Real Forward Pass**: Run model again with computed scalings applied to adapters

This allows the classifier to see the model's internal representations and adaptively weight different expert adapters based on input characteristics.

### Hook-Based Scaling Injection
Uses PyTorch forward hooks to inject computed scalings into each LoRA layer's forward pass without modifying the base model structure.

## Limitations
- **Cache Disabled**: Must set `use_cache=False` because X-LoRA needs to inspect hidden states
- **DoRA Not Supported**: X-LoRA currently doesn't support LoRA layers with DoRA (will raise error)
- **Transformer Only**: Currently only works with transformer architectures

## Reference
Paper: https://huggingface.co/papers/2402.07148
