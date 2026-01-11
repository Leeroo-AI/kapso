# File: `unsloth/models/rl.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1443 |
| Classes | `GuidedDecodingParams` |
| Functions | `vLLMSamplingParams`, `PatchRL`, `patch_functions`, `patch_trl_rl_trainers`, `patch_trl_openenv`, `PatchFastRL` |
| Imports | importlib, inspect, os, re, rl_replacements, torch, trl, typing, unsloth_zoo |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements comprehensive patches for Reinforcement Learning trainers (GRPO, PPO, DPO) from the TRL library, enabling Unsloth optimizations during RL fine-tuning workflows. Provides PatchFastRL function and vLLM sampling parameter compatibility.

**Mechanism:** Main function PatchFastRL() patches TRL trainer classes by: 1) wrapping unwrap_model_for_generation to call FastLanguageModel.for_inference(), 2) patching Trainer.prediction_step with unsloth_prediction_step for proper logit handling (sets UNSLOTH_RETURN_LOGITS=1), 3) dynamically modifying trainer __init__ methods via create_new_function to inject RL_EXTRA_ARGS, RL_FUNCTIONS, RL_PRE_ITEMS from rl_replacements.py, 4) patching generation methods to use .clone() for Unsloth's inference_mode tensors. Includes vLLM compatibility shim (GuidedDecodingParams) and vLLMSamplingParams wrapper. Uses torch_compile_options for optimization.

**Significance:** Critical for RL-based alignment techniques (RLHF, GRPO, DPO) which are increasingly popular for instruction tuning. At 1443 lines, this is one of the largest and most complex modules. The dynamic function patching shows advanced metaprogramming to integrate with TRL without forking. Essential for users doing preference optimization or reward modeling with Unsloth's speed benefits.
