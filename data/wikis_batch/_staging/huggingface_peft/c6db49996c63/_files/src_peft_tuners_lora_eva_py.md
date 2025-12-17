# File: `src/peft/tuners/lora/eva.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 739 |
| Classes | `_Hook`, `SVDHook`, `HashHook` |
| Functions | `find_equal_values`, `get_device_with_meta_params`, `move_inputs_to_device`, `prepare_model_inputs_fn_language_modeling`, `prepare_layer_inputs_fn_language_modeling`, `forward_fn_dict`, `get_eva_state_dict`, `initialize_lora_eva_weights` |
| Imports | collections, config, contextlib, copy, functools, itertools, layer, peft, torch, tqdm, ... +3 more |

## Understanding

**Status:** ✅ Documented

**Purpose:** EVA (Eigenvalue-based Adaptation) - SVD-based initialization using activation statistics

**Mechanism:** Hooks into model forward passes to collect input activations, computes SVD on activation-weighted weight matrices, and initializes LoRA matrices using top singular vectors/values. Supports both exact SVD and hashing-based approximations for memory efficiency. The get_eva_state_dict function orchestrates the entire calibration and initialization process.

**Significance:** Sophisticated initialization method that analyzes actual activation patterns to determine optimal LoRA initialization. Provides significant convergence speedup and improved final performance by starting from a principled decomposition of activation-informed weight updates rather than random initialization.
