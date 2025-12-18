# File: `src/peft/tuners/prompt_tuning/model.py`

**Category:** tuner implementation

| Property | Value |
|----------|-------|
| Lines | 106 |
| Classes | `PromptEmbedding` |
| Imports | math, torch, peft.utils.integrations.gather_params_ctx, transformers.AutoTokenizer (conditional), .config.PromptTuningInit |

## Understanding

**Status:** Explored

**Purpose:** Implements the PromptEmbedding module for Prompt Tuning, which directly learns continuous soft prompt embeddings that are prepended to input sequences. This is the simplest form of prompt learning.

**Mechanism:**
- `PromptEmbedding`: PyTorch module that stores and returns trainable prompt embeddings

  Initialization (behavior depends on `prompt_tuning_init`):

  **Base structure:**
  - Creates embedding layer: `nn.Embedding(total_virtual_tokens, token_dim)`
  - `total_virtual_tokens` = `num_virtual_tokens * num_transformer_submodules`

  **RANDOM initialization** (default):
  - Uses standard PyTorch embedding initialization
  - Warning: May create embeddings outside the natural embedding manifold

  **SAMPLE_VOCAB initialization** (training mode only):
  - Samples random token IDs from vocabulary: `randint(0, vocab_size, total_virtual_tokens)`
  - Looks up corresponding embeddings from `word_embeddings`
  - Clones and detaches these embeddings as initialization
  - Ensures prompts start within the embedding manifold

  **TEXT initialization** (training mode only):
  - Loads tokenizer from `tokenizer_name_or_path`
  - Security: Removes `trust_remote_code` from tokenizer_kwargs to prevent code execution
  - Tokenizes `prompt_tuning_init_text` to get token IDs
  - Handles length mismatch:
    - If text too long: Truncate to `total_virtual_tokens`
    - If text too short: Repeat text until reaching `total_virtual_tokens`
  - Looks up embeddings for these token IDs
  - Uses these semantic embeddings as initialization

  Special handling:
  - Uses `gather_params_ctx` for FSDP compatibility (fully sharded data parallel)
  - Converts embeddings to float32 for training stability
  - Only applies initialization in training mode (not inference)

  Forward pass:
  - Input: Virtual token indices of shape `(batch_size, total_virtual_tokens)`
  - Output: Prompt embeddings of shape `(batch_size, total_virtual_tokens, token_dim)`
  - Simple embedding lookup, no transformations

**Significance:** Core implementation of Prompt Tuning, the foundational prompt learning method. Key characteristics:
1. **Simplicity**: Just trainable embeddings, no encoder networks
2. **Efficiency**: Minimal parameters (only the prompt embeddings)
3. **Initialization matters**: TEXT initialization often performs best by providing semantic grounding
4. **Scales well**: Performance improves with larger models (matches full fine-tuning at scale)

Unlike P-Tuning (which uses MLP/LSTM encoder) or Prefix Tuning (which modifies attention), this directly optimizes continuous embeddings prepended to inputs. The initialization strategy is crucial because gradients may struggle to move randomly initialized embeddings into meaningful regions. TEXT initialization addresses this by starting from embeddings of semantically relevant text.

Used extensively in research and production for parameter-efficient adaptation of large language models.
