# File: `WebAgent/WebWatcher/infer/scripts_eval/mmrag_r1/llm_agent/tensor_helper.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 75 |
| Classes | `TensorConfig`, `TensorHelper` |
| Imports | dataclasses, torch, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides tensor manipulation utilities for managing token sequences, padding, attention masks, and position IDs during LLM generation.

**Mechanism:** Two components: (1) `TensorConfig` dataclass holds pad_token_id, max_prompt_length, max_obs_length, and max_start_length parameters. (2) `TensorHelper` class provides methods: `cut_to_effective_len()` - trims tensors to actual content length based on attention mask; `convert_pad_structure()` - reorganizes padding from right to left or vice versa; `create_attention_mask()` - generates binary mask where non-pad tokens are 1; `create_position_ids()` - computes cumulative positions for transformer attention; `concatenate_with_padding()` - joins tensors and adjusts padding; `_example_level_pad()` - pads responses for inactive examples in a batch.

**Significance:** Essential utility for the generation pipeline. Handles the complex tensor operations needed when managing variable-length sequences across batch processing, multi-turn generation, and GPU padding requirements. Ensures proper handling of padding tokens throughout the generation loop.
