# File: `src/peft/tuners/lora/arrow.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 476 |
| Classes | `ArrowLoraLinearLayer` |
| Functions | `check_loaded_lora_compatibility_arrow`, `ensure_adapters_target_linear_layers_only`, `create_arrow_model` |
| Imports | __future__, config, os, torch, transformers, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Arrow routing for LoRA

**Mechanism:** Implements Arrow (Adaptive Routing with LoRA ensembles) for mixture-of-experts over LoRA adapters. ArrowLoraLinearLayer computes prototype vectors from adapter weights via SVD, then routes tokens to top-k adapters based on cosine similarity during forward passes. build_prototypes() extracts right singular vectors as adapter signatures. forward() computes token-prototype similarities, applies softmax over top-k, and aggregates adapter outputs with weighted sum. gen_know_sub() implements general knowledge subtraction to purify task-specific adapters using task arithmetic. create_arrow_model() loads multiple adapters and configures routing.

**Significance:** Enables dynamic multi-task learning with LoRA by intelligently routing different inputs to different specialized adapters. Arrow makes it possible to efficiently combine multiple LoRA experts in a single model, with per-token routing that adapts to input content. Critical for multi-domain and multi-task scenarios where different regions of input space benefit from different adaptations.
