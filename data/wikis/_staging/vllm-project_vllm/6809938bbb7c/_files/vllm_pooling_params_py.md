# File: `vllm/pooling_params.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 230 |
| Classes | `PoolingParams` |
| Imports | copy, msgspec, typing, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Configuration parameters for embedding and classification models.

**Mechanism:** `PoolingParams` is a msgspec Struct defining parameters for pooling-based models (embeddings, classification, scoring). Key parameters: `dimensions` for matryoshka embeddings, `normalize` for L2 normalization, `use_activation` for classification output activation, `truncate_prompt_tokens` for sequence truncation, `step_tag_id` and `returned_token_ids` for step pooling. Includes validation via `verify()` method that checks task compatibility, merges defaults from model config, and validates parameter combinations. Different tasks (embed/classify/score) support different parameter subsets defined in `valid_parameters`.

**Significance:** Enables vLLM to serve embedding and classification models, not just text generation. Parameters control output format and quality for different use cases. Matryoshka support allows flexible embedding dimensions. The validation ensures users only specify appropriate parameters for their task. This extends vLLM's capabilities beyond LLM generation to the broader transformer model ecosystem.
