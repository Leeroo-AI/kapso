# src/peft/tuners/xlora/classifier.py

## Overview
Implementation of the X-LoRA classifier, a neural network that learns to predict optimal scaling weights for multiple LoRA adapters based on the model's hidden states. This is the "gating" mechanism that enables mixture-of-experts behavior.

## Helper Class: TemperatureScaledSoftmax

A simple module that applies softmax with temperature scaling.

### Initialization
**Parameters:**
- **temperature** (float, default=1.0): Softmax temperature parameter

### Forward Method
Applies temperature-scaled softmax: `softmax(logits / temperature)`

**Effect of Temperature:**
- temperature < 1.0: Sharper probability distribution (more confident)
- temperature = 1.0: Standard softmax
- temperature > 1.0: Softer probability distribution (more uniform)

---

## Main Class: XLoraClassifier

Neural network classifier that predicts per-adapter scaling weights.

### Initialization

**Parameters:**
- **model**: Reference to the PeftModel
- **config**: XLoraConfig with classifier settings
- **n_classes**: Number of LoRA adapters (experts)
- **n_layers**: Number of LoRA adapter layers in the model
- **device**: Device to place classifier on

**Architecture Construction:**

**Single Layer (xlora_depth=1):**
- Direct linear layer from hidden_size to output
- Output size depends on `layerwise_scalings`:
  - True: `n_classes * n_layers` (unique scalings per layer)
  - False: `n_classes` (broadcasted to all layers)
- No bias

**Multi-Layer (xlora_depth > 1):**
1. Input layer: `Linear(hidden_size, xlora_size)`
2. ReLU activation
3. Optional dropout (if `xlora_dropout_p > 0`)
4. Hidden layers: Repeat `xlora_depth - 2` times:
   - `Linear(xlora_size, xlora_size)`
   - ReLU activation
   - Optional dropout
5. Output layer: Same sizing logic as single-layer case

**Attributes:**
- **layers**: nn.Sequential containing full classifier network
- **softmax**: TemperatureScaledSoftmax with configured temperature
- **log_scalings**: List to store scaling predictions (if logging enabled)
- **scalings_logging**: Boolean flag to control logging
- **override_scaling_pass_value**: Value used for dummy scalings during first forward pass
- **dtype**: Dtype to use for classifier (matches base model)

### Core Methods

#### make_dummy_scalings()
Creates dummy scaling tensor for the initial forward pass.

**Parameters:**
- **input_ids** or **inputs_embeds**: Used to infer batch size, sequence length, and device

**Returns:** Tensor of shape `(batch_size, seq_len, n_layers, n_classes)` filled with `override_scaling_pass_value`

**Purpose:** During X-LoRA's first forward pass, these dummy scalings are used to get hidden states without meaningful adapter contributions.

#### forward()
Main forward pass that predicts scaling weights.

**Process:**
1. Extract shape information from input_ids or inputs_embeds
2. Get last hidden state from model output: `hidden_states[-1]`
3. Run classifier network on hidden states to get logits
4. Handle layerwise vs. broadcasted scalings:
   - If not layerwise: Expand logits to all layers via unsqueeze and expand
5. Reshape to `(batch_size, seq_len, n_layers, n_classes)`
6. Apply softmax if `enable_softmax` is True
7. Log scalings if `scalings_logging` is enabled
8. Return scalings tensor

**Input Format:**
- **result**: Output from base model containing hidden_states
- **input_ids** or **inputs_embeds**: For shape inference

**Output Shape:** `(batch_size, seq_len, n_layers, n_classes)`
- Each element represents the predicted weight for a specific (batch, position, layer, adapter) combination

#### _get_bucketed_scalings()
Organizes logged scalings by sequence length for easier analysis.

**Returns:** Dictionary mapping sequence lengths to tuples of (positions, tensors)
- **positions**: List of indices in the original log
- **tensors**: List of scaling tensors with that sequence length

**Purpose:** Different inputs may have different sequence lengths. Bucketing by seq_len allows for batched processing of logged scalings.

#### _set_override_scaling_pass_value()
Updates the value used for dummy scalings.

**Parameters:**
- **value**: Float value to use, or None to set to `1/n_classes`

**Effect:** Updates both the internal value and the config

---

## Design Rationale

### Why Predict from Hidden States?
The classifier sees the model's internal representations at each position, allowing it to make context-aware decisions about which adapters to use. This enables:
- Task-specific adapter selection
- Input-specific adapter weighting
- Dynamic adaptation based on semantic content

### Layerwise vs. Broadcasted Scalings
**Broadcasted** (layerwise_scalings=False):
- Single set of scalings used for all layers
- More parameter efficient
- Assumes similar adapter utility across layers

**Layerwise** (layerwise_scalings=True):
- Unique scalings for each layer
- More expressive but more parameters
- Allows layer-specific adapter specialization

### Two Output Modes
1. **Dense**: All adapters receive non-zero weights (scaled by softmax)
2. **Sparse (top-k)**: Only top-k adapters get non-zero weights (handled in layer code)

### Softmax Application
- Applied after logit prediction if `enable_softmax=True`
- Ensures weights sum to approximately 1.0
- Temperature controls confidence of predictions

### Logging System
The classifier can log all scaling predictions for:
- Analysis of adapter usage patterns
- Debugging
- Visualization
- Model interpretation

Logged scalings are stored as-is without copying, so they reflect the state at prediction time.

## Integration
The classifier is instantiated by `XLoraModel` during initialization and called during the first forward pass to compute scalings that are then injected into all X-LoRA layers via hooks.
