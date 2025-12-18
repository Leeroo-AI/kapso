# src/peft/tuners/trainable_tokens/model.py

## Overview
Implementation of the Trainable Tokens model class that wraps embedding layers to enable selective token training. This class handles the injection of TrainableTokensLayer wrappers and manages weight-tied embeddings.

## Class: TrainableTokensModel

Inherits from `BaseTuner` and provides selective token training functionality.

### Class Attributes

- **prefix**: "trainable_tokens_" - Prefix used for parameter names
- **tuner_layer_cls**: TrainableTokensLayer - Layer wrapper class

---

### Core Methods

#### _prepare_adapter_config()
Prepares the adapter configuration, automatically detecting embedding layer if needed.

**Parameters:**
- **peft_config**: Configuration object
- **model_config**: Model's configuration

**Process:**
1. Check if `target_modules` is None
2. If None: Use `_get_input_embeddings_name()` to detect embedding layer
3. Fallback to `'embed_tokens'` if detection fails
4. Update peft_config.target_modules with detected name

**Purpose:** Enables automatic embedding layer detection for most transformer architectures.

#### inject_adapter()
Injects TrainableTokensLayer adapters into the model, including handling weight tying.

**Parameters:**
- **model**: The base model
- **adapter_name**: Name for the adapter
- **autocast_adapter_dtype** (bool, default=True): Whether to autocast adapter dtype
- **low_cpu_mem_usage** (bool, default=False): Whether to use low CPU memory
- **kwargs**: Additional arguments

**Process:**

**Phase 1: Regular Injection**
1. Call parent's inject_adapter to wrap target embedding layer

**Phase 2: Weight Tying Handling**
2. Get model configuration
3. Find modules tied with embedding (e.g., LM head)
4. Check if `tie_word_embeddings=True` in config
5. Check if input embeddings are now TrainableTokensLayer
6. If conditions met:
   - Iterate through tied weight modules
   - Create tied TrainableTokensLayer for each
   - Set `tied_adapter` parameter to reference main embedding adapter

**Why Handle Weight Tying?**
Many language models tie embeddings between:
- Input embedding layer
- Output prediction head (LM head)

Updates to input embeddings must propagate to output head for consistency. TrainableTokensModel ensures this by:
- Detecting tied weights
- Creating tied adapter for output head
- Tied adapter shares weights with main adapter

#### _get_tied_target_modules()
Override that returns empty list to suppress warnings.

**Purpose:**
Normally, PEFT warns about tied weights that might need special handling. TrainableTokensModel explicitly supports tied weights, so it overrides this to suppress unnecessary warnings.

#### _create_and_replace_dict()
Internal method for creating and replacing modules with adapters, accepting dictionary config.

**Parameters:**
- **peft_config**: Dictionary of configuration (not PeftConfig object)
- **adapter_name**: Name for the adapter
- **target**: Module to wrap
- **target_name**: Name of the target
- **parent**: Parent module
- **current_key**: Full key path

**Process:**
1. Use config dict as kwargs
2. If target is already TrainableTokensLayer:
   - Update with new adapter via `update_layer()`
3. If target is regular module:
   - Create new TrainableTokensLayer
   - Replace original with wrapper

**Why Dict Version?**
Allows adding parameters not in PeftConfig (like `tied_adapter`) for internal use during weight tying setup.

#### _create_and_replace()
Public interface for creating and replacing modules.

**Process:**
1. Convert PeftConfig to dictionary
2. Call `_create_and_replace_dict()` with dictionary

**Purpose:** Provides standard interface while internally using dict-based implementation for flexibility.

#### _create_new_module()
Static method that creates a new TrainableTokensLayer.

**Parameters:**
- **peft_config**: Configuration (as dict)
- **adapter_name**: Name for adapter
- **target**: Module to wrap
- **kwargs**: Additional parameters

**Process:**
1. Create TrainableTokensLayer wrapping target
2. Call `update_layer()` with:
   - init_weights setting
   - token_indices list
   - tied_adapter reference (if applicable)
3. Return new module

**Returns:** Configured TrainableTokensLayer instance

---

## Weight Tying Architecture

### Problem
Language models often tie embeddings:
```
Input: tokens → Embedding Layer → model → Output Layer
                     ↑__________________________|
                            (weight tying)
```

### Solution
TrainableTokensModel detects and handles this:

**Detection:**
1. Check `model_config.tie_word_embeddings`
2. Identify tied weight module names
3. Verify input embeddings wrapped with TrainableTokensLayer

**Handling:**
1. Create tied TrainableTokensLayer for output module
2. Set `tied_adapter` parameter to reference input adapter
3. Tied adapter doesn't create own parameters
4. Instead, uses parameters from main adapter

**Result:**
- Updates to input embeddings automatically affect output head
- No duplication of trainable parameters
- Maintains model's weight tying invariant

---

## Helper Method: _get_module_names_tied_with_embedding()

This method (likely inherited from BaseTuner) identifies modules that share weights with the embedding layer. Common examples:
- `lm_head` in causal language models
- `decoder.embed_tokens` in encoder-decoder models
- Model-specific tied weight configurations

---

## Usage Example

### Basic Usage
```python
from peft import TrainableTokensConfig, get_peft_model

config = TrainableTokensConfig(
    token_indices=[100, 200, 300],
    init_weights=True,
)

model = get_peft_model(base_model, config)
# Automatically detects embedding layer
# Handles weight tying if present
```

### Manual Target Specification
```python
config = TrainableTokensConfig(
    token_indices=new_token_ids,
    target_modules=["model.embed_tokens"],  # Explicit
)

model = get_peft_model(base_model, config)
```

### Adding New Tokens
```python
# 1. Add tokens to tokenizer
tokenizer.add_tokens(["<new1>", "<new2>"])

# 2. Resize model embeddings
model.resize_token_embeddings(len(tokenizer))

# 3. Get new token indices
new_indices = [
    tokenizer.convert_tokens_to_ids("<new1>"),
    tokenizer.convert_tokens_to_ids("<new2>"),
]

# 4. Configure and apply
config = TrainableTokensConfig(
    token_indices=new_indices,
    init_weights=False,  # New tokens, random init
)
peft_model = get_peft_model(model, config)
```

---

## Integration Points

### Embedding Layer Detection
Uses `_get_input_embeddings_name()` utility which:
- Tries `model.get_input_embeddings()` method
- Falls back to common patterns
- Returns appropriate module name

### Weight Tying Detection
Uses model configuration and module inspection to:
- Identify tied weights
- Determine if tying is enabled
- Find all tied modules

### Parameter Tracking
Properly integrates with PEFT's parameter tracking:
- Trainable parameters correctly counted
- State dicts include adapter parameters
- Loading/saving works transparently

---

## Design Decisions

### Why Tied Adapters?
Rather than creating duplicate trainable parameters for tied weights:
- **Efficiency**: No parameter duplication
- **Consistency**: Updates automatically propagate
- **Memory**: Saves memory and optimizer states
- **Correctness**: Maintains model invariants

### Why Dict-Based Internal API?
Using dictionary for `_create_and_replace_dict()`:
- Allows adding parameters not in config
- Enables `tied_adapter` parameter for internal use
- Provides flexibility without modifying config class
- Keeps public API clean

### Why Override _get_tied_target_modules()?
Trainable Tokens explicitly supports and handles tied weights:
- No need for warnings
- Proper handling already implemented
- Suppress unnecessary noise
- User experience improvement

---

## Limitations and Considerations

### FSDP/DeepSpeed Support
Config notes indicate FSDP/DeepSpeed might not be fully supported. Potential issues:
- Distributed parameter handling
- Gradient synchronization
- State dict sharding

### Module Detection
Automatic detection assumes:
- Model has standard embedding structure
- `get_input_embeddings()` method exists
- Or embedding layer named `'embed_tokens'`

May fail for unusual architectures - use manual `target_modules` in such cases.

### Weight Tying Assumptions
Weight tying detection assumes:
- Config has `tie_word_embeddings` field
- Tied modules follow standard naming
- May not catch unusual tying configurations

---

## Integration
This class is instantiated by PEFT's `get_peft_model()` when using TrainableTokensConfig. Handles all the complexity of:
- Embedding layer detection
- Adapter injection
- Weight tying management
- Parameter tracking

Users typically don't interact with this class directly.
