# Implementation: GPTQ LoRA Linear Layer

## File Location
`src/peft/tuners/lora/gptq.py`

## Overview
This module implements LoRA (Low-Rank Adaptation) for GPTQ (Generative Pre-trained Transformer Quantization) quantized linear layers. It supports both AutoGPTQ and GPTQModel libraries and includes support for QA-LoRA (Quantization-Aware LoRA).

## Key Components

### Class: `GPTQLoraLinear`
Applies LoRA adaptation to GPTQ quantized linear layers with optional QA-LoRA support.

**Inheritance:**
- `torch.nn.Module`
- `LoraLayer`

**Key Features:**
- Compatible with both GPTQModel and AutoGPTQ libraries
- Supports standard LoRA and QA-LoRA variants
- Does not support DoRA (raises ValueError if enabled)
- Includes variant system for different LoRA implementations
- Handles group-based quantization for QA-LoRA

**Constructor Parameters:**
- `base_layer`: The GPTQ quantized base layer
- `adapter_name` (str): Name of the adapter
- `r` (int): Rank of the LoRA decomposition (default: 0)
- `lora_alpha` (int): Scaling factor for LoRA (default: 1)
- `lora_dropout` (float): Dropout probability (default: 0.0)
- `init_lora_weights` (bool): Whether to initialize LoRA weights (default: True)
- `use_rslora` (bool): Use rank-stabilized LoRA (default: False)
- `use_dora` (bool): Use DoRA - not supported, must be False
- `use_qalora` (bool): Use Quantization-Aware LoRA (default: False)
- `lora_bias` (bool): Whether to include bias in LoRA (default: False)
- `qalora_group_size` (int): Group size for QA-LoRA quantization (default: 32)

**Attributes:**
- `quant_linear_module`: Reference to base quantized layer

### Method: `resolve_lora_variant(use_dora, use_qalora, **kwargs)`
Determines which LoRA variant to use based on configuration.

**Returns:**
- `DoraLinearVariant` if use_dora=True
- `QALoraLinearVariant` if use_qalora=True
- `None` for vanilla LoRA
- Raises `NotImplementedError` if both dora and qalora requested

**Variant Types:**
1. **Vanilla LoRA**: Standard low-rank adaptation
2. **DoRA**: Weight-decomposed low-rank adaptation (not yet supported)
3. **QA-LoRA**: Quantization-aware LoRA with grouped quantization

### Method: `forward(x: torch.Tensor)`
Performs forward pass with LoRA adaptation on GPTQ quantized weights.

**Implementation Flow:**
1. Execute GPTQ quantized base layer
2. Return early if adapters disabled
3. For each active adapter:
   - Cast input to appropriate dtype
   - Check if adapter has variant (DoRA/QA-LoRA)
   - **Vanilla LoRA**: `result += lora_B(lora_A(dropout(x))) * scaling`
   - **Variant LoRA**: Delegate to variant's forward method
   - Convert result back to original dtype

**Dtype Management:**
- Always casts input to lora_A weight dtype
- Preserves original result dtype for output
- No autocast check (always converts)

### Function: `dispatch_gptq(target, adapter_name, **kwargs)`
Factory function supporting both GPTQModel and AutoGPTQ libraries.

**Parameters:**
- `target` (torch.nn.Module): Target layer to potentially wrap
- `adapter_name` (str): Name for the adapter
- `**kwargs`: Additional configuration (includes `gptq_quantization_config`)

**Returns:**
- New `GPTQLoraLinear` instance if target is GPTQ quantized, None otherwise

**Dual Library Support:**
1. **GPTQModel** (checked first):
   - Uses `gptqmodel.nn_modules.qlinear.BaseQuantLinear`
   - Modern library for GPTQ quantization

2. **AutoGPTQ** (fallback):
   - Uses `get_auto_gptq_quant_linear(cfg)` to get appropriate class
   - Supports various GPTQ implementations
   - Requires quantization config from kwargs

**Logic:**
1. Extracts base layer if target is already a tuner layer
2. Retrieves GPTQ quantization config from kwargs
3. First attempts GPTQModel path if available
4. Falls back to AutoGPTQ with config-based class lookup
5. Creates wrapper and sets `qweight` attribute

## Dependencies
- `torch`
- `peft.import_utils.is_gptqmodel_available`
- `peft.tuners.lora.layer.LoraLayer`
- `peft.tuners.tuners_utils.BaseTunerLayer`
- `peft.utils.get_auto_gptq_quant_linear`
- `peft.tuners.lora.layer.LoraVariant`
- `peft.tuners.lora.variants.DoraLinearVariant` (conditional)
- `peft.tuners.lora.variants.QALoraLinearVariant` (conditional)
- `gptqmodel.nn_modules.qlinear.BaseQuantLinear` (conditional)

## Key Characteristics

### QA-LoRA Support
Unique feature enabling quantization-aware LoRA:
- Uses grouped quantization for LoRA adapters
- Configurable group size (default: 32)
- Further reduces memory footprint
- Variant system allows clean implementation separation

### Variant System
Extensible design pattern:
- Base class handles common logic
- Variants implement specialized behavior
- Clean separation of concerns
- Easy to add new LoRA variants

### Library Compatibility
Smart dispatching supports:
- **GPTQModel**: Modern, actively maintained
- **AutoGPTQ**: Legacy support with multiple backends
- Graceful fallback between libraries
- Config-based class resolution for AutoGPTQ

### Limitations
- **DoRA Not Supported**: Raises `ValueError` if `use_dora=True`
- **DoRA + QA-LoRA**: Raises `NotImplementedError` if both requested
- **No Merging**: Merging not supported (commented in code)

## Usage Context
This adapter is automatically dispatched when:
1. GPTQModel or AutoGPTQ library is installed
2. Target layer is a GPTQ quantized linear layer
3. LoRA configuration is applied to the model

The dispatcher's dual-library support ensures broad compatibility across different GPTQ implementations.

## Notes
- The `__repr__` method prefixes output with "lora." for identification
- Logic differs from default Linear because merging is not supported
- Commented TODO suggests potential Xavier uniform initialization alternative
- QA-LoRA group size parameter passed through to variant
- dtype conversion happens unconditionally (no autocast check)
