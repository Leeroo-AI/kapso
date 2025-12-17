# Implementation: AWQ LoRA Linear Layer

## File Location
`src/peft/tuners/lora/awq.py`

## Overview
This module implements LoRA (Low-Rank Adaptation) for AWQ (Activation-aware Weight Quantization) quantized linear layers. It enables fine-tuning of models quantized with the AWQ method while maintaining the memory efficiency of quantization.

## Key Components

### Class: `AwqLoraLinear`
Applies LoRA adaptation to AWQ quantized linear layers.

**Inheritance:**
- `torch.nn.Module`
- `LoraLayer`

**Key Features:**
- Compatible with AutoAWQ quantization library (version 0.2.0+)
- Does not support DoRA (raises ValueError if enabled)
- Maintains reference to quantized module for backward compatibility
- Handles dtype conversions for quantized operations

**Constructor Parameters:**
- `base_layer`: The AWQ quantized base layer
- `adapter_name` (str): Name of the adapter
- `r` (int): Rank of the LoRA decomposition (default: 0)
- `lora_alpha` (int): Scaling factor for LoRA (default: 1)
- `lora_dropout` (float): Dropout probability (default: 0.0)
- `init_lora_weights` (bool): Whether to initialize LoRA weights (default: True)
- `use_rslora` (bool): Use rank-stabilized LoRA (default: False)
- `use_dora` (bool): Use DoRA - not supported, must be False
- `lora_bias` (bool): Whether to include bias in LoRA (default: False)

**Attributes:**
- `quant_linear_module`: Reference to base layer (for backward compatibility)
- `base_layer`: Same as quant_linear_module (for consistency)

### Method: `forward(x: torch.Tensor)`
Performs forward pass with LoRA adaptation on AWQ quantized weights.

**Implementation Flow:**
1. Compute result through AWQ quantized base layer
2. Return early if adapters are disabled
3. For each active adapter:
   - Retrieve LoRA parameters (lora_A, lora_B, dropout, scaling)
   - Handle dtype conversion when not in autocast mode
   - Apply LoRA transformation: `lora_B(lora_A(dropout(x))) * scaling`
   - Convert output to expected dtype if needed
   - Add to base layer result

**Dtype Handling:**
- Checks if autocast is enabled to determine conversion needs
- Casts input to LoRA weight dtype when required
- Ensures final output matches expected dtype

### Function: `dispatch_awq(target, adapter_name, **kwargs)`
Factory function to create AWQ LoRA adapter with version validation.

**Parameters:**
- `target` (torch.nn.Module): Target layer to potentially wrap
- `adapter_name` (str): Name for the adapter
- `**kwargs`: Additional configuration parameters

**Returns:**
- New `AwqLoraLinear` instance if conditions met, None otherwise

**Version Check:**
- Requires AutoAWQ version >= 0.2.0
- Raises `ImportError` if version is incompatible
- Only validates version at dispatch level (not import)

**Logic:**
1. Extracts base layer if target is already a tuner layer
2. Checks if AutoAWQ is available
3. Verifies target is `WQLinear_GEMM` from AWQ library
4. Validates AutoAWQ version meets minimum requirement
5. Creates `AwqLoraLinear` wrapper if all checks pass
6. Sets `qweight` attribute to reference base layer's quantized weights

## Dependencies
- `torch`
- `packaging.version` - For version comparison
- `importlib.metadata` - For checking installed package versions
- `peft.import_utils.is_auto_awq_available`
- `peft.tuners.lora.layer.LoraLayer`
- `peft.tuners.tuners_utils.BaseTunerLayer`
- `awq.modules.linear.WQLinear_GEMM` (conditional import)

## Key Characteristics

### Version Requirements
- **Minimum AutoAWQ Version**: 0.2.0
- Version check performed at dispatch time, not import time
- Clear error message if version incompatible

### Limitations
- **DoRA Not Supported**: Raises `ValueError` if `use_dora=True`
- **No Merging**: Merging adapters into quantized weights not explicitly supported

### Compatibility
- Designed for `WQLinear_GEMM` layers from AutoAWQ
- Maintains backward compatibility through `quant_linear_module` attribute
- Works with PEFT's adapter management system

### Memory Efficiency
- Base weights remain quantized with AWQ
- Only LoRA adapters use full precision
- Significant memory savings compared to full fine-tuning

## Usage Context
This adapter is automatically selected when:
1. AutoAWQ library (version >= 0.2.0) is installed
2. Target layer is a `WQLinear_GEMM` quantized layer
3. LoRA configuration is applied to the model

The dispatcher integrates with PEFT's automatic adapter selection to enable seamless LoRA training on AWQ-quantized models.

## Notes
- The `__repr__` method prefixes output with "lora." for identification
- Dual naming (`base_layer` and `quant_linear_module`) ensures compatibility across different code paths
- Version validation prevents subtle bugs from incompatible AutoAWQ versions
