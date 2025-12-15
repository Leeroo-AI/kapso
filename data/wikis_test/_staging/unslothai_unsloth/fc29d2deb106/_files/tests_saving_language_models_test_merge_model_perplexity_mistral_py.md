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

**Purpose:** Validates merged Mistral model quality using perplexity

**Mechanism:** Trains Mistral-7B-v0.3 with QLoRA for 200 steps using Alpaca prompt format, saves merged model, then loads in 4-bit, 8-bit (subprocess), and 16-bit to compute perplexity, comparing quality across quantization levels

**Significance:** Ensures Mistral architecture maintains quality after merge and validates Alpaca formatting works correctly with non-chat template models, testing broader model family compatibility

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
