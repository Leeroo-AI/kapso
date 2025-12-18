# File: `src/peft/tuners/prompt_tuning/config.py`

**Category:** tuner configuration

| Property | Value |
|----------|-------|
| Lines | 92 |
| Classes | `PromptTuningInit` (enum), `PromptTuningConfig` (dataclass) |
| Imports | dataclasses, enum, typing, peft.config.PromptLearningConfig, peft.utils.PeftType |

## Understanding

**Status:** Explored

**Purpose:** Defines the configuration class and initialization strategies for Prompt Tuning, the simplest prompt learning method that directly learns continuous soft prompt embeddings prepended to input sequences.

**Mechanism:**
- `PromptTuningInit`: String enum with three initialization strategies:
  - `TEXT`: Initialize from actual text tokens (semantic initialization)
  - `SAMPLE_VOCAB`: Initialize by randomly sampling from vocabulary embeddings
  - `RANDOM`: Random initialization (warning: may fall outside embedding manifold)

- `PromptTuningConfig`: Configuration dataclass that extends `PromptLearningConfig`:
  - Inherits base prompt learning parameters (num_virtual_tokens, token_dim, etc.)
  - Additional Prompt Tuning specific parameters:
    - `prompt_tuning_init`: Initialization strategy (default: RANDOM)
    - `prompt_tuning_init_text`: Text string for TEXT initialization (optional)
    - `tokenizer_name_or_path`: Tokenizer for TEXT initialization (optional)
    - `tokenizer_kwargs`: Additional tokenizer arguments (optional)

  - `__post_init__()`: Validation logic:
    - Sets `peft_type` to `PeftType.PROMPT_TUNING`
    - If TEXT init: Requires both tokenizer_name_or_path and prompt_tuning_init_text
    - If not TEXT init: Rejects tokenizer_kwargs
    - Ensures configuration consistency

**Significance:** Core configuration for Prompt Tuning method. The initialization strategy is crucial for performance:
- **TEXT initialization** (recommended): Uses embeddings of actual text as starting point, providing semantic grounding. For example, initializing with "Classify the sentiment:" gives the model a meaningful starting point.
- **SAMPLE_VOCAB initialization**: Starts with random real word embeddings, ensuring prompts stay within the embedding manifold.
- **RANDOM initialization**: Fastest but may lead to embeddings in unrealistic regions of the embedding space.

Prompt Tuning is the foundational prompt learning method, simpler than P-Tuning (no encoder) and Prefix Tuning (modifies input, not attention). Despite its simplicity, it has been shown to match full fine-tuning on many tasks when scaled to large models.
