# File: `src/peft/tuners/prefix_tuning/model.py`

**Category:** tuner implementation

| Property | Value |
|----------|-------|
| Lines | 81 |
| Classes | `PrefixEncoder` |
| Imports | torch |
| Based on | THUDM P-tuning-v2 implementation |

## Understanding

**Status:** Explored

**Purpose:** Implements the PrefixEncoder neural network for Prefix Tuning, which generates continuous prefix vectors that are prepended to the keys and values in transformer attention layers.

**Mechanism:**
- `PrefixEncoder`: PyTorch module that encodes virtual tokens into prefix key-value pairs

  Initialization (two modes based on `prefix_projection` flag):

  **Without projection** (default, simpler):
  - Single embedding layer: `nn.Embedding(num_virtual_tokens, num_layers * 2 * token_dim)`
  - Output dimension accounts for:
    - `num_layers`: Separate prefixes for each transformer layer
    - `2`: Both keys and values in attention
    - `token_dim`: Dimension per key/value vector
  - Direct optimization of prefix embeddings

  **With projection** (MLP reparameterization):
  - Embedding layer: `nn.Embedding(num_virtual_tokens, token_dim)`
  - 2-layer MLP transformer:
    - Layer 1: `token_dim -> encoder_hidden_size` with Tanh activation
    - Layer 2: `encoder_hidden_size -> num_layers * 2 * token_dim`
  - Provides reparameterization benefits (easier optimization)
  - Only used during training (not in inference mode)

  Forward pass:
  - Input: Prefix indices of shape `(batch_size, num_virtual_tokens)`
  - If projection enabled: Embed -> MLP transform
  - If projection disabled: Direct embedding lookup
  - Output: Prefix key-values of shape `(batch_size, num_virtual_tokens, num_layers * 2 * token_dim)`

Architecture rationale:
- Output tensor is reshaped by PEFT framework to `(batch_size, num_virtual_tokens, num_layers, 2, token_dim)`
- Then split and distributed to each layer's attention mechanism as key/value prefixes
- These prefixes attend to regular tokens, allowing the model to condition on learned "soft prompts"

**Significance:** Core implementation of Prefix Tuning method. Unlike P-Tuning which modifies input embeddings, Prefix Tuning directly injects trainable parameters into the attention mechanism at every layer. This is more powerful because:
1. Affects all layers (not just input)
2. Directly modifies attention's key-value pairs
3. Scales efficiently with model depth
4. Often matches full fine-tuning with <0.1% trainable parameters

Based on THUDM's P-tuning-v2 implementation (which unified several prefix/prompt methods). The optional MLP projection helps with optimization but can be disabled for maximum parameter efficiency. This method is particularly effective for conditional generation tasks.
