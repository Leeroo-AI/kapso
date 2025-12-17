# File: `vllm/pooling_params.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 230 |
| Classes | `PoolingParams` |
| Imports | copy, msgspec, typing, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Pooling model parameters

**Mechanism:** Defines PoolingParams class for configuring pooling/embedding models. Parameters include: truncate_prompt_tokens (prompt truncation), dimensions (matryoshka embedding size), normalize (L2 normalization), use_activation (apply activation for classification). Task-specific parameter validation ensures only relevant parameters are used for each task type (embed, classify, score, token_embed, token_classify). Includes parameter merging from model config defaults and verification logic.

**Significance:** User-facing API for controlling pooling model behavior, analogous to SamplingParams for generation. Essential for embedding, classification, and scoring tasks. Validation prevents misuse of parameters for unsupported tasks. Matryoshka dimension support enables efficient embedding truncation for models trained with nested representations.
