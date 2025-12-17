# File: `src/peft/tuners/adaption_prompt/utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 158 |
| Functions | `llama_rotate_half`, `llama_apply_rotary_pos_emb`, `llama_compute_query_states`, `gpt2_compute_query_states`, `is_adaption_prompt_trainable` |
| Imports | inspect, torch, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Utility functions for recomputing query states and rotary embeddings

**Mechanism:** llama_compute_query_states() recomputes Q projections and applies rotary position embeddings (handles both old 4D and new 2D cache formats, position_ids generation for transformers >=4.37.2). gpt2_compute_query_states() extracts queries from c_attn for cross/self attention. is_adaption_prompt_trainable() filters trainable params

**Significance:** Necessary because transformer forward() methods don't return query states - these functions reconstruct them from hidden states for computing attention scores with prompt keys, handling multiple transformers versions and architectural variants
