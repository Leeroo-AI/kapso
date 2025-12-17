# File: `src/peft/tuners/randlora/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 356 |
| Classes | `RandLoraModel` |
| Imports | __future__, _buffer_dict, accelerate, config, layer, math, peft, torch, transformers, tuners_utils, ... +2 more |

## Understanding

**Status:** ✅ Documented

**Purpose:** Main model class that applies RandLoRA adapters by initializing shared random projection bases (randlora_A, randlora_B) and wrapping target layers with per-layer trainable scaling parameters.

**Mechanism:** Uses _pre_injection_hook to initialize shared random bases via _init_randlora_A_randlora_B (dense or sparse variants based on config). The _find_dim method determines the largest layer dimensions to size the shared matrices. Creates BufferDict for A/B with persistent flag controlled by save_projection. Random bases are initialized with Kaiming uniform using deterministic PRNG seed, then normalized by std. _create_and_replace wraps layers with Linear/Linear8bitLt/Linear4bit variants that share the same A/B references.

**Significance:** Orchestrates RandLoRA's unique architecture where random projection bases are shared globally across all adapted layers, with only per-layer scaling factors being trainable. This dramatically reduces parameters compared to LoRA. The deterministic initialization and optional checkpoint saving balance reproducibility with checkpoint size. Paper: https://huggingface.co/papers/2502.00987.
