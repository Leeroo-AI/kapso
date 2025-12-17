# File: `src/peft/tuners/randlora/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 199 |
| Classes | `RandLoraConfig` |
| Imports | dataclasses, peft, typing, warnings |

## Understanding

**Status:** ✅ Documented

**Purpose:** Configuration for RandLoRA adapters, which use shared random projection bases (randlora_A and randlora_B) with per-layer trainable scaling parameters (lambda and gamma).

**Mechanism:** Extends PeftConfig with r (random basis rank, inversely proportional to trainable params), projection_prng_key for deterministic random initialization, save_projection flag to control checkpoint size, sparse/very_sparse flags for ternary sparse bases, randlora_dropout, randlora_alpha (typically 20×r), and standard PEFT parameters. Validates that projection_prng_key is consistent across adapters and warns if save_projection is False.

**Significance:** RandLoRA reduces trainable parameters by sharing random projection matrices across all adapted layers, training only diagonal scaling factors per layer. The sparse variants aim for matmul-free computation. Parameter r is inversely proportional to trainable parameters (lower r = more params). Paper: https://huggingface.co/papers/2502.00987.
