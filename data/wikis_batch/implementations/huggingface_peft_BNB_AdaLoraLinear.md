# Implementation: BitsAndBytes Quantized AdaLoRA Linear Layers

## File Location
`src/peft/tuners/adalora/bnb.py`

## Overview
This module implements AdaLoRA (Adaptive Low-Rank Adaptation) for BitsAndBytes quantized linear layers. It provides two classes for 8-bit and 4-bit quantization, enabling SVD-based adaptive rank selection with quantized base weights.

## Key Components

### Class: `SVDLinear8bitLt`
AdaLoRA implementation for 8-bit quantized linear layers.

**Conditional Definition:**
- Only defined if BitsAndBytes is available (`is_bnb_available()`)

**Inheritance:**
- `torch.nn.Module`
- `AdaLoraLayer`

**Key Features:**
- Uses SVD-based low-rank adaptation with dynamic rank adjustment
- Works with BitsAndBytes 8-bit linear layers (Int8Params)
- Freezes base layer weights during training
- Includes singular value scaling (lora_E parameter)
- Adaptive rank with ranknum normalization

**Constructor Parameters:**
- `base_layer` (torch.nn.Module): BitsAndBytes 8-bit quantized layer
- `adapter_name` (str): Name of the adapter
- `r` (int): Initial rank (default: 0)
- `lora_alpha` (int): Scaling factor (default: 1)
- `lora_dropout` (float): Dropout probability (default: 0.0)
- `init_lora_weights` (bool): Whether to initialize weights (default: True)

**Initialization:**
1. Freezes base layer weights
2. Sets active adapter
3. Calls update_layer to initialize AdaLoRA parameters

### Method: `forward(x: torch.Tensor)` (SVDLinear8bitLt)
Performs forward pass with AdaLoRA adaptation on 8-bit quantized weights.

**Implementation Flow:**
1. Execute base 8-bit quantized layer
2. Return early if adapters disabled
3. For each active adapter:
   - Handle dtype conversion to float32 if needed
   - Apply SVD-based LoRA: `dropout(x) @ (lora_A * lora_E).T @ lora_B.T`
   - Scale by `scaling / ranknum`
   - Convert back to expected dtype
   - Add to result (avoiding inplace operations)

**Key Differences from Standard LoRA:**
- **lora_E**: Element-wise multiplication with lora_A for singular value scaling
- **ranknum**: Adaptive rank normalization (+ 1e-5 for stability)
- **No Inplace**: Uses `result = result + output` (inplace forbidden for MatMul8bitLtBackward)

**Dtype Handling:**
- Converts to float32 when not in autocast mode
- Required for 8-bit quantized operations

### Class: `SVDLinear4bit`
AdaLoRA implementation for 4-bit quantized linear layers.

**Conditional Definition:**
- Only defined if BitsAndBytes 4-bit is available (`is_bnb_4bit_available()`)

**Inheritance:**
- `torch.nn.Module`
- `AdaLoraLayer`

**Key Features:**
- Same SVD-based approach as 8-bit version
- Works with BitsAndBytes 4-bit linear layers (Params4bit)
- Defensive cloning for 4-bit backward compatibility
- Adaptive rank selection

**Constructor:**
Identical to `SVDLinear8bitLt` with same parameters.

### Method: `forward(x: torch.Tensor, *args, **kwargs)` (SVDLinear4bit)
Forward pass with AdaLoRA adaptation on 4-bit quantized weights.

**Implementation Flow:**
1. Execute base 4-bit quantized layer
2. Return early if adapters disabled
3. **Clone result** (defensive for 4-bit backprop)
4. For each active adapter:
   - Cast input to adapter dtype
   - Apply SVD-based LoRA computation
   - Scale by `scaling / ranknum`
   - Convert output to expected dtype
   - Add to result

**4-Bit Specific:**
- **Defensive Clone**: `result = result.clone()`
- Prevents backprop errors on manipulated views
- May be resolved in newer PyTorch versions

**Dtype Handling:**
- Uses `_cast_input_dtype` helper method
- More sophisticated than 8-bit version

## Dependencies
- `torch`
- `typing.Any`
- `peft.import_utils.{is_bnb_available, is_bnb_4bit_available}`
- `peft.tuners.adalora.layer.AdaLoraLayer`

## Key Characteristics

### AdaLoRA Specifics

**SVD-Based Adaptation:**
- **lora_A**: Left singular vectors
- **lora_B**: Right singular vectors
- **lora_E**: Singular values (element-wise with lora_A)
- **ranknum**: Adaptive rank tracking

**Mathematical Formulation:**
```
output = dropout(x) @ (lora_A * lora_E).T @ lora_B.T
scaled_output = output * scaling / (ranknum + 1e-5)
```

**Adaptive Rank:**
- `ranknum` tracks effective rank
- Normalized with small epsilon (1e-5) for stability
- Enables dynamic rank adjustment during training

### Quantization Integration

**8-Bit Integration:**
- Works with `bnb.nn.Linear8bitLt`
- Uses `Int8Params` for weights
- Requires float32 conversion for operations
- No inplace operations (MatMul8bitLtBackward constraint)

**4-Bit Integration:**
- Works with `bnb.nn.Linear4bit`
- Uses `Params4bit` for weights
- Requires defensive cloning for backprop safety
- More memory efficient than 8-bit

### Limitations
- **No Merge/Unmerge**: Not supported for quantized layers (yet)
- **Frozen Base Weights**: Base layer always frozen during training
- **Float32 Required**: Operations need float32 conversion

### Memory Efficiency
- Base weights in quantized format (8-bit or 4-bit)
- Only AdaLoRA parameters (A, B, E) in full precision
- Adaptive rank reduces parameter count further

## Usage Context
These classes are used when:
1. BitsAndBytes library is installed
2. Model loaded with 8-bit or 4-bit quantization
3. AdaLoRA configuration applied

They enable adaptive rank LoRA on memory-constrained hardware.

## Notes
- Both classes use `__repr__` prefix "adalora." for identification
- The 4-bit defensive clone comment references Tim Dettmers (BitsAndBytes author)
- Inplace operation warning specific to 8-bit backward pass
- Small epsilon (1e-5) prevents division by zero in ranknum
- Element-wise multiplication with lora_E is key difference from standard LoRA
- Future versions may support merge/unmerge as commented in code
