# Implementation: BitsAndBytes Quantized IA3 Linear Layers

## File Location
`src/peft/tuners/ia3/bnb.py`

## Overview
This module implements IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations) for BitsAndBytes quantized linear layers. It provides two classes for 8-bit and 4-bit quantization, enabling parameter-efficient fine-tuning through learned scaling vectors.

## Key Components

### Class: `Linear8bitLt`
IA3 implementation for 8-bit quantized linear layers.

**Conditional Definition:**
- Only defined if BitsAndBytes is available (`is_bnb_available()`)

**Inheritance:**
- `torch.nn.Module`
- `IA3Layer`

**Key Features:**
- Uses learned scaling vectors instead of low-rank matrices
- Works with BitsAndBytes 8-bit quantization
- Supports feedforward and attention layers differently
- Freezes base layer weights
- Minimal parameter overhead

**Constructor Parameters:**
- `base_layer` (torch.nn.Module): BitsAndBytes 8-bit quantized layer
- `adapter_name` (str): Name of the adapter
- `is_feedforward` (bool): Whether layer is feedforward or attention
- `init_ia3_weights` (bool): Whether to initialize IA3 weights (default: True)

**Key Attribute:**
- `is_feedforward`: Determines where scaling is applied (input vs output)

**Initialization:**
1. Freezes base layer weights
2. Sets active adapter
3. Calls update_layer to initialize IA3 parameters

### Method: `forward(x: torch.Tensor, *args, **kwargs)` (Linear8bitLt)
Performs forward pass with IA3 scaling on 8-bit quantized weights.

**Implementation Flow:**
1. Return base layer output if adapters disabled
2. Compute combined scaling from all active adapters
3. Handle dtype conversion to float32 if needed
4. Apply scaling based on layer type:
   - **Feedforward**: Scale input, then base layer
   - **Attention**: Base layer, then scale output
5. Convert result back to expected dtype

**IA3 Scaling Logic:**
```python
ia3_scaling = 1
for active_adapter in self.active_adapters:
    ia3_scaling *= self.ia3_l[active_adapter].flatten()

if self.is_feedforward:
    result = base_layer(x * ia3_scaling)  # Input scaling
else:
    result = base_layer(x) * ia3_scaling  # Output scaling
```

**Multiplicative Combination:**
Multiple adapters multiply their scalings together.

### Class: `Linear4bit`
IA3 implementation for 4-bit quantized linear layers.

**Conditional Definition:**
- Only defined if BitsAndBytes 4-bit is available (`is_bnb_4bit_available()`)

**Inheritance:**
- `torch.nn.Module`
- `IA3Layer`

**Key Features:**
- Same IA3 approach as 8-bit version
- Works with 4-bit quantization
- Includes defensive cloning for backward compatibility
- Supports feedforward/attention distinction

**Constructor:**
Identical to `Linear8bitLt` with same parameters.

### Method: `forward(x: torch.Tensor, *args, **kwargs)` (Linear4bit)
Forward pass with IA3 scaling on 4-bit quantized weights.

**Implementation Flow:**
Same as 8-bit version with additional defensive cloning:

1. Return base layer output if adapters disabled
2. Compute combined ia3_scaling
3. Handle dtype conversion to float32 if needed
4. Apply scaling (feedforward or attention style)
5. **Clone result** (defensive for 4-bit)
6. Convert to expected dtype

**4-Bit Specific:**
```python
result = result.clone()
```

**Rationale:**
- Duplicated from adalora.py and lora.py
- For 4-bit training on older PyTorch versions
- Prevents backprop errors on manipulated views
- May be resolved in newer PyTorch

## Dependencies
- `torch`
- `typing.Any`
- `peft.import_utils.{is_bnb_available, is_bnb_4bit_available}`
- `peft.tuners.ia3.layer.IA3Layer`

## Key Characteristics

### IA3 Method

**Core Concept:**
Instead of low-rank matrices, IA3 uses learned scaling vectors:
- **Feedforward layers**: Scale inputs before transformation
- **Attention layers**: Scale outputs after transformation

**Parameter Efficiency:**
- Only learns a single scaling vector (ia3_l)
- Much fewer parameters than LoRA
- Vector size matches in_features or out_features

**Mathematical Formulation:**
- **Feedforward**: `y = W(x * s)` where s is learned scaling
- **Attention**: `y = (Wx) * s` where s is learned scaling

### Adapter Composition
**Multiplicative Combination:**
When multiple adapters are active:
```python
ia3_scaling = scaling_1 * scaling_2 * ... * scaling_n
```

This allows composing multiple adaptations.

### Quantization Integration

**8-Bit Integration:**
- Works with `bnb.nn.Linear8bitLt`
- Requires float32 conversion
- No inplace operation issues (compared to LoRA)

**4-Bit Integration:**
- Works with `bnb.nn.Linear4bit`
- Requires defensive cloning
- More memory efficient

### Dtype Handling
Both versions:
1. Check if autocast is enabled
2. Convert to float32 if needed
3. Apply operations
4. Convert back to expected dtype

**Conversion Logic:**
```python
requires_conversion = (not torch.is_autocast_enabled()) and (x.dtype != torch.float32)
if requires_conversion:
    x = x.float()
```

### Limitations
- **No Merge/Unmerge**: Not supported for quantized layers (yet)
- **Frozen Base Weights**: Base layer always frozen
- **Float32 Required**: Operations need float32 conversion

## Usage Context
These classes are used when:
1. BitsAndBytes library is installed
2. Model loaded with 8-bit or 4-bit quantization
3. IA3 configuration applied

Provides ultra-parameter-efficient adaptation for quantized models.

## Notes
- Both classes use `__repr__` prefix "ia3." for identification
- Feedforward vs attention distinction is core to IA3 method
- Simpler than LoRA (no matrix multiplications, just element-wise scaling)
- Defensive clone comment duplicated from adalora.py and lora.py
- Multiplicative adapter composition is unique to IA3
- Fewer parameters than LoRA but potentially less expressive
- The flatten() call ensures scaling vector broadcasts correctly
