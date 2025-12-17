# Implementation: AQLM LoRA Linear Layer

## File Location
`src/peft/tuners/lora/aqlm.py`

## Overview
This module implements LoRA (Low-Rank Adaptation) for AQLM (Accurate Quantized Low-rank Matrix) quantized linear layers. It provides a specialized adapter for applying LoRA to models quantized with the AQLM quantization method.

## Key Components

### Class: `AqlmLoraLinear`
Extends LoRA functionality to work with AQLM quantized linear layers.

**Inheritance:**
- `torch.nn.Module`
- `LoraLayer`

**Key Features:**
- Does not support DoRA (raises ValueError if enabled)
- Merging is not supported for AQLM quantized layers
- Handles dtype conversions for quantized weights
- Applies LoRA adaptation on top of AQLM quantization

**Constructor Parameters:**
- `base_layer`: The AQLM quantized base layer
- `adapter_name` (str): Name of the adapter
- `r` (int): Rank of the LoRA decomposition (default: 0)
- `lora_alpha` (int): Scaling factor for LoRA (default: 1)
- `lora_dropout` (float): Dropout probability (default: 0.0)
- `init_lora_weights` (bool): Whether to initialize LoRA weights (default: True)
- `use_rslora` (bool): Use rank-stabilized LoRA (default: False)
- `use_dora` (bool): Use DoRA - not supported, must be False
- `lora_bias` (bool): Whether to include bias in LoRA (default: False)

### Method: `forward(x: torch.Tensor)`
Performs forward pass with LoRA adaptation applied to AQLM quantized weights.

**Implementation Details:**
1. Calls base AQLM quantized layer first
2. Returns early if adapters are disabled
3. For each active adapter:
   - Retrieves lora_A, lora_B, dropout, and scaling parameters
   - Handles dtype conversion when not in autocast mode
   - Applies LoRA computation: `lora_B(lora_A(dropout(x))) * scaling`
   - Converts output back to expected dtype if needed
   - Adds result to base layer output

**Dtype Handling:**
- Checks if autocast is enabled
- Converts input to LoRA weight dtype if needed
- Ensures output matches expected dtype

### Function: `dispatch_aqlm(target, adapter_name, **kwargs)`
Factory function to create AQLM LoRA adapter when appropriate.

**Parameters:**
- `target` (torch.nn.Module): Target layer to potentially wrap
- `adapter_name` (str): Name for the adapter
- `**kwargs`: Additional configuration parameters

**Returns:**
- New `AqlmLoraLinear` instance if target is AQLM quantized, None otherwise

**Logic:**
1. Extracts base layer if target is already a tuner layer
2. Checks if AQLM is available and target is `QuantizedLinear`
3. Creates `AqlmLoraLinear` wrapper if conditions met
4. Sets `qweight` attribute to point to base layer's codes

## Dependencies
- `torch`
- `peft.import_utils.is_aqlm_available`
- `peft.tuners.lora.layer.LoraLayer`
- `peft.tuners.tuners_utils.BaseTunerLayer`
- `aqlm.QuantizedLinear` (conditional import)

## Key Characteristics

### Limitations
- **DoRA Not Supported**: Raises `ValueError` if `use_dora=True`
- **No Merging**: Unlike standard LoRA, merging adapters into base weights is not supported for AQLM
- **Forward-Only Adaptation**: Adapters can only be applied during forward pass

### Dtype Management
The implementation carefully manages dtype conversions to ensure compatibility between:
- AQLM quantized weights (typically int-based)
- LoRA adapter weights (typically float)
- Input activations
- Output activations

### Memory Efficiency
- Leverages AQLM's quantization for base weights
- Only LoRA adapters use full precision
- Minimal memory overhead compared to full fine-tuning

## Usage Context
This adapter is automatically dispatched when:
1. AQLM library is installed and available
2. Target layer is an AQLM `QuantizedLinear` layer
3. LoRA configuration is applied to the model

The dispatcher integrates with PEFT's automatic adapter selection system to seamlessly apply LoRA to AQLM-quantized models.

## Notes
- The `__repr__` method prefixes output with "lora." for identification
- Commented TODO suggests potential alternative initialization strategy (Xavier uniform for lora_A, zeros for lora_B)
- Logic differs from standard Linear LoRA specifically because merging is not supported
