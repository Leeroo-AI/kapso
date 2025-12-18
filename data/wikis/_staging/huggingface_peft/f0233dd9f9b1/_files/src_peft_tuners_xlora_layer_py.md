# src/peft/tuners/xlora/layer.py

## Overview
Implementation of X-LoRA layer wrappers that intercept LoRA layer forward passes and apply dynamic scaling from the X-LoRA classifier. These layers are the core mechanism for applying mixture-of-experts style adapter weighting.

## Class Hierarchy

### Base Class: XLoraLayer

Abstract base class that wraps any LoRA layer and provides X-LoRA functionality.

#### Initialization
**Parameters:**
- **model**: Reference to the XLoraModel instance
- **target**: The original LoRA layer being wrapped
- **target_forward**: Original forward method of the target layer
- **layer_number**: Index of this layer in the model
- **config**: XLoraConfig object

#### Core Methods

**apply_scalings_to_x(x, scalings_layer, adapter)** (static)
Applies per-adapter scaling to input tensor.

**Process:**
1. Extract scaling for specific adapter from scalings tensor
2. Scalings shape: [batch_size, seq_len, n_classes]
3. Select adapter column: [batch_size, seq_len, 1]
4. Element-wise multiply input by scaling

**get_maybe_topk_scalings(scalings)**
Processes raw scalings from classifier, optionally applying top-k selection and softmax.

**Process:**
1. Extract scalings for this specific layer from full tensor
2. If `top_k_lora` is set:
   - Select top-k adapter indices
   - Create mask with True for top-k, False for others
   - Zero out non-top-k scalings
3. If `enable_softmax_topk` is True:
   - Apply softmax only to non-zero (top-k) values
   - Preserve zeros for non-selected adapters

**Returns:** Processed scalings tensor [batch_size, seq_len, n_classes]

---

### Class: XLoraLinearLayer

X-LoRA wrapper for `lora.Linear` layers (fully connected layers).

#### Forward Method
**Signature:** `forward(x, *args, scalings=None, **kwargs)`

**Process:**
1. Store original dtype for restoration
2. If scalings provided, process them with `get_maybe_topk_scalings()`
3. Run base layer forward (without LoRA)
4. For each active adapter:
   - Skip if merged or adapter doesn't exist
   - Check DoRA not used (not supported)
   - Cast input to adapter dtype
   - Apply scalings to input if provided
   - Compute LoRA contribution: `lora_B(lora_A(dropout(scaled_x))) * scaling * global_weight`
   - Add to result
5. Restore original dtype
6. Return result

**Key Features:**
- Supports multiple adapters simultaneously
- Applies global_scaling_weight when scalings provided
- Preserves dtype throughout computation

---

### Class: XLoraEmbeddingLayer

X-LoRA wrapper for `lora.Embedding` layers (token embeddings).

#### Forward Method
**Signature:** `forward(x, *args, scalings=None, **kwargs)`

**Process:**
1. Process scalings if provided
2. Run base layer forward (without LoRA)
3. Get embedding scaling factor (for models like Gemma that scale embeddings)
4. For each active adapter:
   - Skip if merged or adapter doesn't exist
   - Check DoRA not used (not supported)
   - Apply embedding_A to input
   - Apply scalings if provided
   - Multiply by embedding_B and scaling factor
   - Apply embed_scale if present (maintains consistency with base layer)
   - Add to result
5. Return result

**Special Handling:**
- Properly handles models that scale embeddings in their forward method
- Uses `_get_embed_scale()` to match base layer behavior

---

### Class: XLoraConv2dLayer

X-LoRA wrapper for `lora.Conv2d` layers (2D convolutional layers).

#### Forward Method
**Signature:** `forward(x, *args, scalings=None, **kwargs)`

**Process:**
1. Store original dtype
2. Process scalings if provided
3. Run base layer forward (without LoRA)
4. For each active adapter:
   - Skip if merged or adapter doesn't exist
   - Check DoRA not used (not supported)
   - Cast input to adapter dtype
   - Apply scalings to input if provided
   - Compute LoRA contribution: `lora_B(lora_A(dropout(scaled_x))) * scaling * global_weight`
   - Add to result
5. Restore original dtype
6. Return result

**Implementation:** Nearly identical to Linear layer but for convolutional operations

---

## Key Design Patterns

### Scaling Application
All layers follow the same pattern for applying scalings:
1. Check if scalings are provided (None during non-X-LoRA forward)
2. Process scalings (apply top-k, softmax if configured)
3. Apply per-adapter scaling before LoRA computation
4. Use global_scaling_weight when scalings are active

### Adapter Iteration
All layers iterate through active adapters and:
- Skip if merged (adapters already integrated into base weights)
- Skip if adapter not in LoRA dictionaries
- Validate DoRA not used (raise error if found)
- Apply adapter-specific transformations

### Dtype Management
All layers preserve dtype consistency:
- Store original input dtype
- Cast to adapter dtype during computation
- Restore original dtype before returning

## Limitations

### DoRA Not Supported
All layers explicitly check and raise ValueError if DoRA (Weight-Decomposed Low-Rank Adaptation) is used with X-LoRA. This is a current limitation that may be addressed in future versions.

### Scalings Requirement
Scalings must be provided as keyword argument in forward pass. This is handled automatically by X-LoRA's hook system but must be considered for custom implementations.

## Integration
These layers are instantiated and injected by `convert_layers_to_xlora()` in the model file. They replace the forward methods of existing LoRA layers to intercept and apply dynamic scaling.
