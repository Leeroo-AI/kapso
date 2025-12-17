# File: `src/peft/tuners/cpt/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 99 |
| Classes | `CPTConfig` |
| Imports | dataclasses, peft, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines configuration parameters for CPT (Context-aware Prompt Tuning) prompt learning method.

**Mechanism:** CPTConfig dataclass extends PromptLearningConfig with CPT-specific parameters: cpt_token_ids (token IDs for prompts), cpt_mask (mask for tokens), cpt_tokens_type_mask (token type indicators), loss weighting options (opt_weighted_loss_type, opt_loss_decay_factor), projection epsilons for input and format, and tokenizer path. Validates that all mask arrays match length in __post_init__.

**Significance:** Core configuration for context-aware prompt tuning, exclusively supporting TaskType.CAUSAL_LM. Paper reference: https://huggingface.co/papers/2410.17222. Enables sophisticated prompt optimization with token-level control, weighted losses, and projection-based regularization.
