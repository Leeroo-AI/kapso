# File: `src/peft/tuners/lora/eva.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 739 |
| Classes | `_Hook`, `SVDHook`, `HashHook` |
| Functions | `find_equal_values`, `get_device_with_meta_params`, `move_inputs_to_device`, `prepare_model_inputs_fn_language_modeling`, `prepare_layer_inputs_fn_language_modeling`, `forward_fn_dict`, `get_eva_state_dict`, `initialize_lora_eva_weights` |
| Imports | collections, config, contextlib, copy, functools, itertools, layer, peft, torch, tqdm, ... +3 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** EVA initialization with hooks

**Mechanism:** Implements EVA (Eigenvalue-based Variance Adaptation) initialization using forward hooks to capture layer inputs during inference. SVDHook and HashHook classes intercept activations during model forward passes. initialize_lora_eva_weights() orchestrates the process: runs model on calibration data, captures layer inputs via hooks, computes SVD on weight matrices, and initializes LoRA weights based on singular vectors. Supports both IPM (Important Parameter Matching) and KPM (Keep Parameter Matching) strategies. get_eva_state_dict() extracts computed singular values/vectors for caching. Handles various model types through forward_fn_dict dispatch.

**Significance:** Provides data-driven LoRA initialization that accounts for actual model usage patterns. By analyzing activations on representative data, EVA finds low-rank subspaces that preserve important model behaviors. More sophisticated than static weight-based initialization, especially valuable when fine-tuning domain differs from pretraining.
