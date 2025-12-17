# Implementation: EETQ LoRA Linear Layer

## File Location
`src/peft/tuners/lora/eetq.py`

## Overview
This module implements LoRA (Low-Rank Adaptation) for EETQ (Easy and Efficient Quantization Toolkit) quantized linear layers. It enables parameter-efficient fine-tuning of models quantized with EETQ while maintaining quantization benefits.

## Key Components

### Class: `EetqLoraLinear`
Applies LoRA adaptation to EETQ quantized linear layers.

**Conditional Definition:**
- Only defined if EETQ is available (`is_eetq_available()`)
- Imported within conditional block to avoid import errors

**Inheritance:**
- `torch.nn.Module`
- `LoraLayer`

**Key Features:**
- Works with EETQ's int8 quantization
- Does not support DoRA (raises ValueError if enabled)
- Explicitly disables merge/unmerge operations (raises AttributeError)
- Maintains backward compatibility reference to quantized module

**Constructor Parameters:**
- `base_layer`: The EETQ quantized base layer
- `adapter_name` (str): Name of the adapter
- `r` (int): Rank of the LoRA decomposition (default: 0)
- `lora_alpha` (int): Scaling factor for LoRA (default: 1)
- `lora_dropout` (float): Dropout probability (default: 0.0)
- `init_lora_weights` (bool): Whether to initialize LoRA weights (default: True)
- `use_rslora` (bool): Use rank-stabilized LoRA (default: False)
- `use_dora` (bool): Use DoRA - not supported, must be False
- `lora_bias` (bool): Whether to include bias in LoRA (default: False)

**Attributes:**
- `quant_linear_module`: Reference to base layer for backward compatibility

### Method: `forward(x: torch.Tensor)`
Performs forward pass with LoRA adaptation on EETQ quantized weights.

**Implementation Flow:**
1. Execute EETQ quantized base layer
2. Return early if adapters disabled
3. For each active adapter:
   - Retrieve LoRA components (lora_A, lora_B, dropout, scaling)
   - Handle dtype conversion when not in autocast mode
   - Apply LoRA: `lora_B(lora_A(dropout(x))) * scaling`
   - Convert output to expected dtype if needed
   - Add to base layer result

**Dtype Conversion Logic:**
- Checks autocast status to determine conversion needs
- Casts input to match lora_A weight dtype
- Ensures output matches expected result dtype

### Method: `merge(safe_merge, adapter_names)`
**Explicitly Not Supported** - Raises `AttributeError`

**Error Message:**
"Merging LoRA layers is not supported for Eetq layers."

### Method: `unmerge()`
**Explicitly Not Supported** - Raises `AttributeError`

**Error Message:**
"Unmerging LoRA layers is not supported for Eetq layers."

### Function: `dispatch_eetq(target, adapter_name, **kwargs)`
Factory function to create EETQ LoRA adapter when appropriate.

**Parameters:**
- `target` (torch.nn.Module): Target layer to potentially wrap
- `adapter_name` (str): Name for the adapter
- `**kwargs`: Additional configuration parameters

**Returns:**
- New `EetqLoraLinear` instance if target is EETQ quantized, None otherwise

**Logic:**
1. Extracts base layer if target is already a tuner layer
2. Checks if EETQ is available and target is `EetqLinear`
3. Creates `EetqLoraLinear` wrapper if conditions met
4. Sets `weight` attribute to reference base layer weights
5. Optionally sets `bias` attribute if present in base layer

## Dependencies
- `torch`
- `peft.import_utils.is_eetq_available`
- `peft.tuners.lora.layer.LoraLayer`
- `peft.tuners.tuners_utils.BaseTunerLayer`
- `eetq.EetqLinear` (conditional import)

## Key Characteristics

### Limitations
- **DoRA Not Supported**: Raises `ValueError` if `use_dora=True`
- **No Merge/Unmerge**: Explicitly blocks merging operations with clear error messages
- **Forward-Only Adaptation**: Adapters can only be applied during forward pass

### Design Rationale
The explicit blocking of merge/unmerge operations suggests:
- EETQ quantization format doesn't support weight modification
- Maintaining quantized representation is critical for EETQ efficiency
- Safer to error early than produce incorrect results

### Memory Efficiency
- Base weights remain in EETQ quantized format (int8)
- LoRA adapters stored in full precision
- Minimal overhead compared to full model fine-tuning

### Weight Handling
Unlike some other quantized adapters, this implementation:
- Directly references base layer's `weight` attribute
- Also handles bias if present in base layer
- No separate `qweight` attribute needed for EETQ

## Usage Context
This adapter is automatically dispatched when:
1. EETQ library is installed and available
2. Target layer is an `EetqLinear` quantized layer
3. LoRA configuration is applied to the model

The conditional class definition ensures no import errors occur when EETQ is not installed.

## Notes
- The `__repr__` method prefixes output with "lora." for identification
- Dual naming convention (`base_layer` and `quant_linear_module`) maintains consistency with other quantized adapters
- Explicit error messages in merge/unmerge help users understand limitations
- Bias handling is optional and checked with `hasattr`
