# File: `tests/saving/language_models/test_merge_model_perplexity_llama-3.2.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 259 |
| Functions | `formatting_prompts_func`, `load_and_compute_8bit_ppl` |
| Imports | datasets, gc, multiprocessing, pandas, pathlib, sys, tests, torch, tqdm, transformers, ... +2 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** Explored

**Purpose:** Perplexity benchmark test that validates model quality is preserved through the QLoRA training and merge process for Llama-3.2-3B-Instruct.

**Mechanism:** Uses multiprocessing for memory isolation when loading models in different quantization modes. Computes perplexity on openassistant-guanaco-reformatted eval split across 5 configurations: (1) base model 4-bit, (2) QLoRA-trained model, (3) merged model loaded in 4-bit, (4) merged model loaded in 8-bit (subprocess), (5) merged model loaded in 16-bit. Trains with LoRA rank 16, 10 training steps, using `train_on_responses_only` for response-focused learning. Uses `ppl_model()` and comparison tracking utilities from tests.utils.perplexity_eval.

**Significance:** Validates that Unsloth's merge process preserves model quality across different quantization levels for the Llama-3.2 architecture. The perplexity comparison helps detect any quality degradation from merging or quantization artifacts.
