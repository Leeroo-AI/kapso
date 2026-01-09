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

**Purpose:** Phi-4 perplexity benchmark that ensures Unsloth's merge pipeline works with Microsoft's Phi architecture.

**Mechanism:** Trains Phi-4 model with rank-16 LoRA for 200 steps using phi-4 chat template and train_on_responses_only with Phi-specific tokens (<|im_start|>user<|im_sep|>, <|im_start|>assistant<|im_sep|>). Measures perplexity across 5 stages: base 4-bit, LoRA, merged 4-bit, merged 8-bit (subprocess), merged 16-bit.

**Significance:** Validates Unsloth supports the Phi architecture which uses different attention mechanisms (partial rotary embeddings) and different chat format than LLaMA/Mistral. Critical for ensuring Unsloth works beyond Meta/Mistral models with architecturally distinct designs from Microsoft Research.
