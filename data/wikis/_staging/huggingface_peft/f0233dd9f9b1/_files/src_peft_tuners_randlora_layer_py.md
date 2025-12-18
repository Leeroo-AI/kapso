# File: `src/peft/tuners/randlora/layer.py`

**Category:** layer

| Property | Value |
|----------|-------|
| Lines | 351 |
| Classes | `UniqueBaseGrad`, `RandLoraLayer`, `Linear` |
| Imports | _buffer_dict, peft, torch, transformers, typing, warnings |

## Understanding

**Status:** ✅ Fully explored

**Purpose:** Implements RandLora adapter layers with frozen random bases and trainable diagonal scaling matrices.

**Mechanism:**
- **UniqueBaseGrad class** (torch.autograd.Function): Memory-efficient gradient computation for unique shared bases
  - Forward: Computes `out = lambda[:,:,None] * randlora_A * gamma[None,]` (element-wise scaling)
  - Backward: Custom gradients for lambda and gamma only (A is frozen):
    - `grad_lambda = einsum("kbj,kvj,bj->kb", grad_out, A, gamma)`
    - `grad_gamma = einsum("kbj,kvj,kb->bj", grad_out, A, lambda)`
  - Returns None gradient for randlora_A (frozen)

- **RandLoraLayer class** (base layer):
  - **Trainable Parameters:**
    - `randlora_lambda`: (r, num_bases) scaling coefficients
    - `randlora_gamma`: (num_bases, min_dim) scaling vector
  - **Non-trainable References:**
    - `randlora_A`: Optional[BufferDict] - reference to shared basis A
    - `randlora_B`: Optional[BufferDict] - reference to shared basis B
  - `adapter_layer_names = ("randlora_lambda", "randlora_gamma")`
  - `merged_adapters`: List tracking merged adapters
  - `update_layer()`: Initializes lambda/gamma parameters and validates base dimensions
  - `reset_randlora_parameters()`: Zeros lambda, sets gamma to 1/max(dims)

- **Linear class** (concrete implementation):
  - Inherits from `nn.Linear` and `RandLoraLayer`
  - **Key Methods:**
    - `get_scaled_bases()`: Scales and slices shared bases for this layer's dimensions
      1. Slices randlora_A/B to match layer dimensions
      2. Applies UniqueBaseGrad for memory-efficient scaling
      3. Flattens over rank and num_bases dimensions
      4. Returns bases in correct order (smallest dimension first)
    - `get_delta_weight()`: Computes full delta weight matrix for merging
      - Formula: `(update_B @ update_A) * scaling`
      - Handles fan_in_fan_out transpose
    - `forward()`: Applies RandLora adaptation
      1. Base layer forward pass
      2. For each adapter: `result += F.linear(F.linear(x, B), A) * scaling`
      3. Uses dropout and proper dtype casting
    - `merge()`: Merges adapter weights into base layer (with safe_merge option)
    - `unmerge()`: Removes merged adapter weights from base layer

- **Computation Flow:**
  - Shared bases (frozen): randlora_A and randlora_B
  - Per-layer scaling: lambda and gamma (trainable)
  - Combined bases: `A_scaled = lambda * A * gamma`
  - Forward: `x @ A_scaled @ B_scaled * alpha/r`

**Significance:** Core implementation of RandLora's parameter-efficient design. Key innovations:
1. **Shared frozen random bases**: Massive parameter reduction compared to LoRA
2. **Trainable diagonal scaling**: Per-layer adaptation with minimal parameters
3. **Memory-efficient gradients**: Custom autograd for unique base reduces memory usage
4. **Slicing mechanism**: Shared bases sized for largest layer, sliced for smaller layers
The design achieves similar performance to LoRA with ~10x fewer trainable parameters by exploiting the fact that random projections can be shared across layers when combined with layer-specific scaling.
