# File: `tests/saving/language_models/test_merge_model_perplexity_llama-3.2.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 259 |
| Functions | `formatting_prompts_func`, `load_and_compute_8bit_ppl` |
| Imports | datasets, gc, multiprocessing, pandas, pathlib, sys, tests, torch, tqdm, transformers, ... +2 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** LLaMA 3.2 perplexity benchmark that measures model quality degradation (if any) across different quantization levels after merge.

**Mechanism:** Trains LLaMA 3.2-3B with rank-16 LoRA for 10 steps on OpenAssistant dataset, computes perplexity at 5 checkpoints: (1) base 4-bit model, (2) LoRA-adapted model, (3) merged model reloaded in 4-bit, (4) merged model in 8-bit (subprocess), (5) merged model in 16-bit. Uses train_on_responses_only to mask instruction tokens during training.

**Significance:** Quantifies the quality preservation of Unsloth's merge operation across quantization formats. The perplexity comparison table reveals if merging introduces numerical errors or if different quantization levels maintain model performance, essential for production deployment decisions.
