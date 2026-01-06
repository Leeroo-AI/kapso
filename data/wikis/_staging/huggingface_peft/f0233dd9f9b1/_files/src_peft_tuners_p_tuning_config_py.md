# File: `src/peft/tuners/p_tuning/config.py`

**Category:** tuner configuration

| Property | Value |
|----------|-------|
| Lines | 61 |
| Classes | `PromptEncoderReparameterizationType` (enum), `PromptEncoderConfig` (dataclass) |
| Imports | dataclasses, enum, typing, peft.config.PromptLearningConfig, peft.utils.PeftType |

## Understanding

**Status:** Explored

**Purpose:** Defines the configuration class and enumerations for P-Tuning, a prompt learning method that uses trainable continuous prompts encoded by a neural network.

**Mechanism:**
- `PromptEncoderReparameterizationType`: String enum with two encoder architectures:
  - `MLP`: Multi-layer perceptron encoder (simpler, faster)
  - `LSTM`: LSTM-based encoder (more expressive, bidirectional)

- `PromptEncoderConfig`: Configuration dataclass that extends `PromptLearningConfig`:
  - Inherits base prompt learning parameters (num_virtual_tokens, token_dim, etc.)
  - Additional P-Tuning specific parameters:
    - `encoder_reparameterization_type`: MLP or LSTM (default: MLP)
    - `encoder_hidden_size`: Hidden dimension of encoder network (required)
    - `encoder_num_layers`: Number of encoder layers (default: 2)
    - `encoder_dropout`: Dropout probability (default: 0.0)
  - `__post_init__()`: Sets `peft_type` to `PeftType.P_TUNING`

**Significance:** Core configuration for P-Tuning method. Unlike simple prompt tuning which learns embeddings directly, P-Tuning learns a neural encoder that generates the prompt embeddings. This reparameterization can lead to better optimization and more expressive prompts. The MLP encoder is recommended for most use cases as it's simpler and faster than LSTM while being effective. This approach is based on the original P-Tuning paper which showed that encoding prompts through a small neural network improves performance over direct embedding optimization.
