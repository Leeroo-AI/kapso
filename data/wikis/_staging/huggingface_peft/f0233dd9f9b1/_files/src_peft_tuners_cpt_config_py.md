# File: `src/peft/tuners/cpt/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 99 |
| Classes | `CPTConfig` |
| Imports | dataclasses, peft, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines CPTConfig, the configuration dataclass for Context-aware Prompt Tuning (CPT), which extends PromptLearningConfig to specify hyperparameters for token masks, loss weighting, and epsilon-based projection constraints.

**Mechanism:** CPTConfig stores three key types of parameters: (1) Token-related (cpt_token_ids, cpt_mask, cpt_tokens_type_mask) that define which tokens are virtual prompts and their types; (2) Loss-related (opt_weighted_loss_type, opt_loss_decay_factor) for controlling exponential decay in loss computation; (3) Projection-related (opt_projection_epsilon, opt_projection_format_epsilon) for constraining the norm of learned prompt embeddings. The __post_init__ method validates that all mask arrays have the same length, sets num_virtual_tokens, and enforces that CPT only works with causal language modeling tasks.

**Significance:** This configuration class is essential for CPT implementation, described in https://huggingface.co/papers/2410.17222. CPT differentiates itself from standard prompt tuning by applying type-specific constraints on prompt tokens (input templates, actual inputs, and output templates) and using epsilon-bounded projections to prevent overfitting. The token type masks enable selective gradient updates, while loss decay helps balance multiple examples in few-shot scenarios.
