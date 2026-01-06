# File: `src/peft/tuners/adaption_prompt/utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 158 |
| Functions | `llama_rotate_half`, `llama_apply_rotary_pos_emb`, `llama_compute_query_states`, `gpt2_compute_query_states`, `is_adaption_prompt_trainable` |
| Imports | inspect, torch, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Utility functions for query recomputation

**Mechanism:** llama_compute_query_states() recomputes query states with rotary position embeddings (handles transformers version differences for position_embeddings/rotary_emb API). gpt2_compute_query_states() extracts queries from c_attn projection. is_adaption_prompt_trainable() checks if parameter name starts with "adaption_".

**Significance:** Essential utilities for adaption prompt layer. Query state recomputation is necessary because original forward() doesn't return queries needed for adapter attention computation.
