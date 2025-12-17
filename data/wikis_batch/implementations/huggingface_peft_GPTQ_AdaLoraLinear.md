# Implementation: GPTQ Quantized AdaLoRA Linear Layer

## File Location
`src/peft/tuners/adalora/gptq.py`

## Overview
This module implements AdaLoRA (Adaptive Low-Rank Adaptation) for GPTQ quantized linear layers. It provides SVD-based adaptive rank adaptation for models quantized with GPTQ, enabling parameter-efficient fine-tuning with dynamic rank selection.

## Key Components

### Class: `SVDQuantLinear`
AdaLoRA implementation for GPTQ quantized linear layers.

**Inheritance:**
- `torch.nn.Module`
- `AdaLoraLayer`

**Key Features:**
- SVD-based low-rank adaptation with adaptive rank
- Compatible with GPTQ quantization
- Maintains quantized base weights
- Includes singular value scaling (lora_E)
- Adaptive rank normalization

**Constructor Parameters:**
- `base_layer`: GPTQ quantized base layer
- `adapter_name` (str): Name of the adapter
- `r` (int): Initial rank (default: 0)
- `lora_alpha` (int): Scaling factor (default: 1)
- `lora_dropout` (float): Dropout probability (default: 0.0)
- `init_lora_weights` (bool): Initialize weights (default: True)

**Attributes:**
- `quant_linear_module`: Reference to base layer (backward compatibility)
- `base_layer`: Same as quant_linear_module (consistency)

**Initialization:**
1. Calls AdaLoraLayer.__init__ with base layer
2. Sets up dual naming (quant_linear_module and base_layer)
3. Sets active adapter
4. Calls update_layer with adapter parameters

### Method: `forward(x: torch.Tensor)`
Performs forward pass with AdaLoRA adaptation on GPTQ quantized weights.

**Implementation Flow:**
1. Execute GPTQ quantized base layer
2. Return early if adapters disabled
3. For each active adapter:
   - Retrieve AdaLoRA parameters (lora_A, lora_B, lora_E, dropout, scaling, ranknum)
   - Handle dtype conversion to float32 if not in autocast
   - Apply SVD-based computation: `(dropout(x) @ (lora_A * lora_E).T @ lora_B.T) * scaling / ranknum`
   - Convert entire expression result to expected dtype
   - Add to base result

**Mathematical Formulation:**
```
output = dropout(x) @ (lora_A * lora_E).T @ lora_B.T
scaled_output = output * scaling / (ranknum + 1e-5)
result += scaled_output
```

**Key Components:**
- **lora_A**: Left singular vectors (in_features × r)
- **lora_B**: Right singular vectors (out_features × r)
- **lora_E**: Singular values (element-wise multiplication with lora_A)
- **ranknum**: Adaptive rank tracking (+ 1e-5 for numerical stability)

### Dtype Conversion Pattern
**Unique Characteristic:**
Unlike BitsAndBytes versions, dtype conversion applies to the **entire expression** rather than intermediate results.

**Implementation:**
```python
requires_conversion = not torch.is_autocast_enabled()
if requires_conversion:
    expected_dtype = result.dtype
    x = self._cast_input_dtype(x, torch.float32)

output = (dropout(x) @ (lora_A * lora_E).T @ lora_B.T) * scaling / ranknum

if requires_conversion:
    output = output.to(expected_dtype)  # Entire expression converted
result += output
```

**Note in Code:**
TODO comment questions if this is correct compared to SVDLinear8bitLT and SVDLinear4bit, which convert intermediate results.

### Method: `__repr__()`
Returns string representation with "adalora." prefix.

**Indentation Issue:**
The method has incorrect indentation (inside forward method), suggesting a potential bug in the source file.

## Dependencies
- `torch`
- `peft.tuners.adalora.layer.AdaLoraLayer`

## Key Characteristics

### AdaLoRA Integration
**SVD-Based Decomposition:**
- Uses singular value decomposition for low-rank adaptation
- lora_E stores singular values
- Element-wise multiplication: `lora_A * lora_E`
- Enables adaptive rank during training

**Adaptive Rank:**
- `ranknum` tracks effective rank
- Normalization factor: `ranknum + 1e-5`
- Small epsilon prevents division by zero
- Enables dynamic rank adjustment

### GPTQ Compatibility
**Quantized Base:**
- Works with GPTQ quantized linear layers
- Base weights remain quantized during training
- Only AdaLoRA parameters in full precision

**Dual Naming:**
- `quant_linear_module`: For backward compatibility
- `base_layer`: For consistency with other adapters
- Both reference same quantized layer

### Dtype Management
**Float32 Operations:**
- Casts input to float32 when not in autocast mode
- Required for numerical stability with quantized weights
- Converts entire output expression back to expected dtype

**Difference from BNB:**
- BNB versions convert intermediate dropout(x) result
- This version converts final output expression
- TODO suggests this may need verification

### Limitations
- **No Merge/Unmerge**: Not implemented for GPTQ layers
- **Forward-Only**: Adapters only applied during forward pass
- **Float32 Required**: Operations need float32 for stability

## Usage Context
This class is used when:
1. GPTQ quantization is applied to model
2. AdaLoRA configuration specified
3. Need for adaptive rank selection

Enables memory-efficient fine-tuning with dynamic rank adjustment on GPTQ-quantized models.

## Notes
- The `__repr__` method appears to have indentation bug (inside forward method)
- TODO comment highlights potential difference in dtype conversion pattern
- Small epsilon (1e-5) in ranknum prevents numerical instability
- Element-wise lora_A * lora_E multiplication is key AdaLoRA characteristic
- No explicit dispatch function in this file (likely handled elsewhere)
- Simpler than BitsAndBytes versions (no special backward pass considerations)
