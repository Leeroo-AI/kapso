# File: `src/peft/tuners/vera/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 162 |
| Classes | `VeraConfig` |
| Imports | __future__, dataclasses, peft, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Configuration dataclass for VeRA (Vector-based Random Matrix Adaptation) tuning parameters.

**Mechanism:** Defines VeraConfig with parameters including rank r (default 256, higher than LoRA), projection_prng_key for deterministic initialization, save_projection flag, vera_dropout, d_initial (0.1), target_modules, and standard PEFT options. Validates layers_pattern requires layers_to_transform and warns if projections won't be saved.

**Significance:** Provides configuration interface for VeRA, which uses shared random projections (vera_A, vera_B) with trainable per-layer scaling vectors (lambda_d, lambda_b) for extreme parameter efficiency compared to LoRA. References paper https://huggingface.co/papers/2310.11454.
