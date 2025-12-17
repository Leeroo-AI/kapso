# File: `examples/offline_inference/spec_decode.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 234 |
| Functions | `get_custom_mm_prompts`, `parse_args`, `main` |
| Imports | transformers, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates speculative decoding methods (EAGLE, ngram, MTP)

**Mechanism:** Compares speculative decoding methods: eagle/eagle3 with draft models, ngram with prompt lookup, and MTP (Medusa-style). Collects metrics (spec_decode_num_drafts, num_draft_tokens, num_accepted_tokens) to calculate mean acceptance length and per-position acceptance rates. Supports multimodal inputs.

**Significance:** Example demonstrating various speculative decoding techniques for faster generation with acceptance metrics.
