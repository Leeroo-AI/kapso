# src/peft/tuners/ln_tuning/model.py

## Overview
Implementation of the LN Tuning (LayerNorm Tuning) model class that makes only normalization layers trainable while freezing the rest of the model. This provides an extremely parameter-efficient fine-tuning approach.

## Class: LNTuningModel

Inherits from `BaseTuner` and provides LayerNorm-specific tuning functionality.

### Class Attributes

- **prefix**: "ln_tuning_" - Prefix used for LN tuning parameter names
- **tuner_layer_cls**: LNTuningLayer - Layer wrapper class
- **target_module_mapping**: TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING - Default target modules per architecture

### Methods

#### _create_and_replace()
Creates and replaces target normalization layers with LN tuning wrappers.

**Parameters:**
- **peft_config**: PeftConfig with LN Tuning settings
- **adapter_name**: Name for the adapter
- **target**: The normalization layer to be adapted
- **target_name**: Name of the target layer
- **parent**: Parent module containing the target
- **current_key**: Full key path to the module

**Process:**
1. Create new LNTuningLayer wrapping the target
2. If adapter is not the active adapter: Freeze the new module
3. Replace original module with the wrapper

**Key Behavior:**
- Only active adapters are kept trainable
- Inactive adapters are frozen

#### _create_new_module()
Creates a new LNTuningLayer instance or updates an existing one.

**Parameters:**
- **peft_config**: Configuration object
- **target**: The layer to wrap
- **adapter_name**: Name of the adapter

**Logic:**
- If target is not already LNTuningLayer:
  - Create new LNTuningLayer wrapping target
- If target is already LNTuningLayer:
  - Update existing layer with new adapter
  - Adds additional adapter to the same layer

**Returns:** LNTuningLayer instance

#### _unloading_checks()
Validates that unloading operation is safe.

**Parameters:**
- **adapter_names**: Optional list of adapters to unload

**Validation:**
- Checks if any adapters specify `modules_to_save`
- Raises ValueError if trying to unload multiple adapters with modules_to_save
- This prevents conflicts when merging different task-specific heads

#### _unload_and_optionally_merge()
Unloads adapters and optionally merges them into the base model.

**Parameters:**
- **merge** (bool, default=True): Whether to merge before unloading
- **progressbar** (bool, default=False): Show progress bar
- **safe_merge** (bool, default=False): Validate merge safety
- **adapter_names**: Optional list of specific adapters to unload

**Process:**
1. Run unloading checks
2. Get all module keys (excluding those with prefix)
3. Iterate through modules:
   - Try to get parent, target, and target name
   - If target has base_layer (is wrapped):
     - Optionally merge adapter into base layer
     - Replace wrapper with unwrapped base layer
4. Return modified model

**Purpose:** Clean removal of LN tuning infrastructure when adapters no longer needed

#### _cast_adapter_dtype()
Override of parent method that does nothing for LN Tuning.

**Reason:**
LN Tuning doesn't add new adapter parameters - it creates copies of original layers. Changing dtype of these copies would cause dtype mismatches with the rest of the model.

**Why Override?**
Parent class's default implementation would attempt to cast adapter layers to a specific dtype, which is inappropriate for LN Tuning's architecture where adapters are full copies of the original layers.

---

## Architecture Design

### No New Parameters
Unlike LoRA, LoKr, and similar methods, LN Tuning doesn't add new parameters. Instead:
1. Original normalization layer is kept as `base_layer`
2. A copy is created for each adapter
3. During forward, active adapter's copy is used
4. During training, only the active adapter's copy is updated

### Copy-Based Approach
Each adapter gets a complete copy of the normalization layer:
- **Advantages:**
  - Simple implementation
  - No special forward logic needed
  - Can switch adapters by just swapping which copy is used
- **Disadvantages:**
  - Memory overhead (one copy per adapter)
  - Not as memory-efficient as shared-weight approaches

### Single Adapter Limitation
LN Tuning only allows:
- One active adapter during forward pass
- One merged adapter at a time
- This simplification makes the implementation cleaner and more efficient

---

## Usage Example

From the docstring:

```python
from transformers import AutoModelForCausalLM
from peft import get_peft_model, TaskType, LNTuningConfig

peft_config = LNTuningConfig(
    task_type=TaskType.CAUSAL_LM,
)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

**Expected Output:**
Shows very small percentage of trainable parameters (typically <1% for large models)

---

## Workflow

### Adding an Adapter
1. Identify target normalization layers
2. Wrap each with LNTuningLayer
3. LNTuningLayer creates a copy of the original layer
4. Original layer stored as base_layer
5. Copy stored in ln_tuning_layers dict

### Using an Adapter
1. Set adapter as active
2. Forward pass uses active adapter's copy
3. Gradient updates only affect active adapter
4. Other adapters remain unchanged

### Merging an Adapter
1. Swap base_layer with adapter's copy
2. Adapter's copy becomes the new base
3. Old base stored in adapter's slot
4. Can be unmerged by swapping back

### Removing an Adapter
1. Optionally merge first
2. Replace LNTuningLayer with base_layer
3. Remove wrapper, restore original structure

---

## Integration Points

### Automatic Target Detection
Uses `TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING` to automatically identify normalization layers for common architectures. This mapping knows which layers are normalization layers for:
- BERT-based models
- GPT-based models
- T5-based models
- LLaMA-based models
- And many more

### Multi-Adapter Support
While only one adapter can be active during forward pass, multiple adapters can coexist:
- Different adapters for different tasks
- Switch between them via `set_adapter()`
- Useful for multi-task inference

---

## Limitations

### Single Adapter Forward
Cannot combine multiple LN tuning adapters in a single forward pass:
- LoRA can run multiple adapters simultaneously
- LN Tuning switches between adapters
- This is by design for simplicity

### Memory Overhead
Each adapter requires a full copy of all target normalization layers:
- Not as memory-efficient as parameter-only methods (LoRA, etc.)
- But normalization layers are typically small compared to model size
- Trade-off: simplicity vs. memory efficiency

### No Autocasting
The `_cast_adapter_dtype()` method is disabled because LN Tuning uses full layer copies rather than separate adapter weights. This can cause issues in mixed-precision training if not handled carefully.

---

## Reference
Paper: https://huggingface.co/papers/2312.11420

## Integration
This class is instantiated by PEFT's `get_peft_model()` function when using LNTuningConfig, or can be used directly for more control.
