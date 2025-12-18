# src/peft/tuners/ln_tuning/config.py

## Overview
Configuration class for LN Tuning (LayerNorm Tuning), defining parameters for selecting which normalization layers to make trainable. This is one of the simplest PEFT configurations, with minimal parameters needed.

## Class: LNTuningConfig

Inherits from `PeftConfig` and provides configuration for LayerNorm tuning.

### Configuration Parameters

#### Target Module Selection
- **target_modules** (Optional[Union[list[str], str]], default=None):
  - List of module names or regex patterns to apply LN Tuning to
  - Examples: `'.*decoder.*'`, `'.*encoder.*'`, `['layer_norm', 'final_norm']`
  - If not specified: Uses architecture-specific defaults
  - If architecture unknown: Raises error (must specify manually)

**Common Patterns:**
- `'.*layer_norm.*'`: All LayerNorm layers
- `'.*decoder.*norm.*'`: All normalization in decoder
- `'.*LayerNorm'`: All modules ending with LayerNorm

#### Exclusion
- **exclude_modules** (Optional[Union[list[str], str]], default=None):
  - Module names or regex patterns to exclude from LN Tuning
  - String: Performs regex match
  - List: Performs exact match or suffix match
  - Takes precedence over target_modules

#### Additional Trainable Modules
- **modules_to_save** (Optional[Union[list[str], str]], default=None):
  - Additional modules to be trainable and saved in checkpoint
  - Common use case: Task-specific output heads
  - Examples: `'classifier'`, `'score'`, `'lm_head'`
  - Useful for sequence classification, token classification tasks
  - These layers are typically randomly initialized and need training

### Post-Initialization

The `__post_init__` method:
1. Calls parent's post-init
2. Sets `peft_type` to `PeftType.LN_TUNING`

## Design Philosophy

### Minimal Configuration
LN Tuning has one of the simplest configurations in PEFT:
- No rank parameters (not applicable)
- No alpha/scaling parameters
- No dropout parameters
- Only needs to specify which layers to tune

### Why So Simple?
Unlike methods like LoRA or LoKr that add new parameters, LN Tuning simply:
1. Identifies normalization layers
2. Makes them trainable
3. Keeps everything else frozen

The only configuration needed is identifying which layers to tune.

## Target Module Selection Strategy

### Automatic Selection
If `target_modules=None`, the system attempts to:
1. Detect model architecture
2. Select appropriate normalization layers based on architecture
3. If architecture unknown, raise error

### Manual Selection
When specifying manually, consider:
- **LayerNorm**: Most common in transformers
- **RMSNorm**: Used in some modern architectures (LLaMA, etc.)
- **BatchNorm**: Less common in transformers, more in CNNs
- **GroupNorm**: Some vision or hybrid models

### Best Practices
1. **Start conservative**: Target only LayerNorm initially
2. **Expand if needed**: Add other norm layers if performance insufficient
3. **Use regex carefully**: Ensure pattern matches intended layers
4. **Verify**: Check `model.print_trainable_parameters()` to confirm

## Comparison with Other Methods

### Parameter Efficiency
LN Tuning typically requires:
- **<1%** of model parameters for large models
- Even less than LoRA in most cases
- Only the normalization layer parameters (scale and bias)

### When to Use LN Tuning
**Good for:**
- Extremely limited compute/memory
- Quick domain adaptation
- Multi-task scenarios (small adapters per task)
- When model already has good representations

**Consider alternatives when:**
- Need more expressive adaptation (use LoRA)
- Task very different from pre-training (use larger methods)
- Have sufficient resources for fuller fine-tuning

## Example Usage

### Basic Configuration
```python
from peft import LNTuningConfig, get_peft_model

config = LNTuningConfig(
    task_type="CAUSAL_LM",
    target_modules='.*layer_norm',
)
```

### With Task Head
```python
config = LNTuningConfig(
    task_type="SEQ_CLS",
    target_modules=['.*layer_norm'],
    modules_to_save=['classifier'],  # Train classification head too
)
```

### Architecture-Specific
```python
# For BERT-like models
config = LNTuningConfig(
    task_type="TOKEN_CLS",
    target_modules=['.*LayerNorm'],
    modules_to_save=['classifier'],
)

# For decoder-only models
config = LNTuningConfig(
    task_type="CAUSAL_LM",
    target_modules=['.*norm'],  # Catches LayerNorm, RMSNorm, etc.
)
```

## Typo Note
The docstring has a typo: "shoud" should be "should". This is in the help text for target_modules.

## Integration
This configuration is passed to `get_peft_model()` or used directly with `LNTuningModel` to apply LayerNorm tuning to a base model.

## Reference
Paper: https://huggingface.co/papers/2312.11420
