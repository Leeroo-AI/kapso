# File: `tests/saving/language_models/test_merge_model_perplexity_phi_4.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 259 |
| Functions | `formatting_prompts_func`, `load_and_compute_8bit_ppl` |
| Imports | datasets, gc, multiprocessing, pandas, pathlib, sys, tests, torch, tqdm, transformers, ... +2 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Perplexity validation for Microsoft's Phi-4 model with phi-specific chat template.

**Mechanism:**
- Loads "unsloth/Phi-4" in 4-bit with phi-4 chat template
- Uses response-only training with Phi-4 format markers (`<|im_start|>`, `<|im_sep|>`)
- Trains for 200 steps with LoRA rank 16
- Computes perplexity across all stages: base, QLoRA, merged in 4/8/16-bit
- Subprocess-isolated 8-bit evaluation to prevent memory issues

**Significance:** Tests Unsloth's compatibility with Microsoft's Phi architecture, which differs from Llama/Mistral. Validates chat template handling for Phi's unique format and ensures merge quality preservation across quantization levels for this architecture.
