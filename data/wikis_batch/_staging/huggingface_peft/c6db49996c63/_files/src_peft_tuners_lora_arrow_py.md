# File: `src/peft/tuners/lora/arrow.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 476 |
| Classes | `ArrowLoraLinearLayer` |
| Functions | `check_loaded_lora_compatibility_arrow`, `ensure_adapters_target_linear_layers_only`, `create_arrow_model` |
| Imports | __future__, config, os, torch, transformers, typing |

## Understanding

**Status:** ✅ Documented

**Purpose:** Arrow (Adaptive Router for Rank-Optimized LoRA Weights) - dynamic mixture-of-experts approach for LoRA

**Mechanism:** ArrowLoraLinearLayer implements token-level routing that selects top-k LoRA adapters per token. Uses prototype-based routing (builds representative activation patterns) and GenKnowSub (Generate Knowledge Subspace) to extract knowledge from existing adapters. Routes each token to the most relevant subset of adapters dynamically.

**Significance:** Advanced LoRA variant that enables efficient multi-adapter inference by routing different tokens to different experts. Allows flexible combination of multiple LoRA adapters with per-token specialization, dramatically improving multi-task and multi-domain performance.
