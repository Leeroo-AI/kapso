# File: `tests/saving/language_models/test_merge_model_perplexity_mistral.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 318 |
| Functions | `load_and_compute_8bit_ppl` |
| Imports | datasets, gc, multiprocessing, pandas, pathlib, sys, tests, torch, tqdm, transformers, ... +2 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Perplexity validation for Mistral-7B-v0.3 model using Alpaca prompt format instead of chat templates.

**Mechanism:**
- Identical structure to Llama-3.2 test but uses Mistral-7B and Alpaca instruction format
- Converts OpenAssistant conversations to Alpaca format (Instruction/Input/Response structure)
- Trains for 200 steps (vs 10 in Llama test) due to larger model size
- Tests perplexity across base 4-bit, QLoRA, and merged models loaded in 4/8/16-bit
- Uses multiprocessing for 8-bit perplexity to avoid OOM

**Significance:** Validates Unsloth's merge quality for Mistral architecture and non-chat-template formats. Tests that Alpaca-style formatting works correctly through the training pipeline. Ensures architectural differences (Mistral vs Llama) don't affect merge correctness.
