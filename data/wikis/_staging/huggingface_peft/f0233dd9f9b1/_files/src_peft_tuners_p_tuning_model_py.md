# File: `src/peft/tuners/p_tuning/model.py`

**Category:** tuner implementation

| Property | Value |
|----------|-------|
| Lines | 131 |
| Classes | `PromptEncoder` |
| Imports | torch, warnings, .config |
| Based on | NVIDIA NeMo implementation |

## Understanding

**Status:** Explored

**Purpose:** Implements the PromptEncoder neural network for P-Tuning, which generates continuous prompt embeddings through a trainable encoder network (MLP or LSTM) rather than learning embeddings directly.

**Mechanism:**
- `PromptEncoder`: PyTorch module that encodes virtual tokens into embeddings

  Initialization:
  - Creates embedding layer for virtual tokens: `nn.Embedding(total_virtual_tokens, token_dim)`
  - `total_virtual_tokens` = `num_virtual_tokens * num_transformer_submodules`
  - If not inference mode, builds encoder network based on `encoder_type`:

    **MLP encoder** (recommended):
    - 3-layer MLP: input_size -> hidden_size -> hidden_size -> output_size
    - Uses ReLU activations between layers
    - Warns if encoder_num_layers != 2 (MLP always uses exactly 2 hidden layers)

    **LSTM encoder**:
    - Bidirectional LSTM with configurable layers and dropout
    - Followed by 2-layer MLP: (hidden_size * 2) -> (hidden_size * 2) -> output_size
    - Output size doubled due to bidirectional processing

  Forward pass:
  - Input: Token indices of shape `(batch_size, total_virtual_tokens)`
  - Embeds indices to get initial representations
  - Passes through encoder (MLP or LSTM+MLP)
  - Output: Encoded embeddings of shape `(batch_size, total_virtual_tokens, token_dim)`

Architecture differences:
- MLP: Simple feedforward, faster, recommended
- LSTM: Captures sequential dependencies between virtual tokens, more expressive but slower

**Significance:** Core implementation of P-Tuning method. The key innovation is reparameterizing prompt embeddings through a neural encoder rather than optimizing embeddings directly. This approach:
1. Provides better optimization landscape (easier to train)
2. Enables sharing of information across virtual tokens
3. Can capture dependencies between prompt positions (especially with LSTM)
4. Often achieves better performance than direct prompt tuning

Based on NVIDIA NeMo's implementation, this is a production-grade version of the P-Tuning paper's approach. The MLP encoder is typically sufficient and preferred for efficiency, while LSTM can be used when modeling token dependencies is important.
