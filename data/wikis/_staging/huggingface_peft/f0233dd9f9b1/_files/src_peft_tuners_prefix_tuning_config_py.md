# File: `src/peft/tuners/prefix_tuning/config.py`

**Category:** tuner configuration

| Property | Value |
|----------|-------|
| Lines | 43 |
| Classes | `PrefixTuningConfig` (dataclass) |
| Imports | dataclasses, peft.config.PromptLearningConfig, peft.utils.PeftType |

## Understanding

**Status:** Explored

**Purpose:** Defines the configuration class for Prefix Tuning, a prompt learning method that prepends trainable continuous vectors to the key and value activations in attention layers across all transformer layers.

**Mechanism:**
- `PrefixTuningConfig`: Configuration dataclass that extends `PromptLearningConfig`:
  - Inherits base prompt learning parameters (num_virtual_tokens, token_dim, num_layers, etc.)
  - Additional Prefix Tuning specific parameters:
    - `encoder_hidden_size`: Hidden size of the MLP encoder (default: None, required if prefix_projection=True)
    - `prefix_projection`: Whether to use MLP reparameterization (default: False)
      - If True: Uses 2-layer MLP to encode prefix embeddings
      - If False: Directly learns prefix embeddings (simpler, fewer parameters)
  - `__post_init__()`: Sets `peft_type` to `PeftType.PREFIX_TUNING`

**Significance:** Core configuration for Prefix Tuning method. Unlike P-Tuning which adds trainable tokens to the input embeddings, Prefix Tuning directly modifies the key-value pairs in attention mechanisms across all layers. The `prefix_projection` flag controls whether to use MLP reparameterization:
- Without projection (default): Simpler, directly optimizes prefix embeddings
- With projection: Uses small MLP for reparameterization, can improve optimization and generalization (similar to P-Tuning's approach)

Prefix Tuning is particularly effective for generation tasks and has been shown to match full fine-tuning performance on many benchmarks while only training a tiny fraction of parameters.
