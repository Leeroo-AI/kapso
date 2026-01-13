# File: `tests/saving/language_models/test_merge_model_perplexity_phi_4.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 259 |
| Functions | `formatting_prompts_func`, `load_and_compute_8bit_ppl` |
| Imports | datasets, gc, multiprocessing, pandas, pathlib, sys, tests, torch, tqdm, transformers, ... +2 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** Explored

**Purpose:** Perplexity benchmark test that validates model quality preservation through QLoRA training and merge for Microsoft's Phi-4 model.

**Mechanism:** Loads `unsloth/Phi-4` in 4-bit, applies Phi-4 specific chat template (using `<|im_start|>` and `<|im_sep|>` tokens), trains with LoRA rank 16 for 200 steps. Uses `train_on_responses_only` with Phi-4's specific instruction/response markers. Computes perplexity across base 4-bit, QLoRA, merged 4-bit, merged 8-bit (subprocess), and merged 16-bit configurations. Saves merged model to `./unsloth_out/merged_phi4_text_model`.

**Significance:** Validates Unsloth's support for Microsoft's Phi architecture, which uses a different tokenizer and chat template format than Llama/Mistral models. Ensures the merge process works correctly for this smaller but capable model family.
