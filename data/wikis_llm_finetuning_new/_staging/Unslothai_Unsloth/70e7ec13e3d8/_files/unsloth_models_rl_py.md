# File: `unsloth/models/rl.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1443 |
| Classes | `GuidedDecodingParams` |
| Functions | `vLLMSamplingParams`, `PatchRL`, `patch_functions`, `patch_trl_rl_trainers`, `patch_trl_openenv`, `PatchFastRL` |
| Imports | importlib, inspect, os, re, rl_replacements, torch, trl, typing, unsloth_zoo |

## Understanding

**Status:** Explored

**Purpose:** Comprehensive TRL (Transformer Reinforcement Learning) trainer patching system that modifies SFT, GRPO, DPO, and other trainers for Unsloth compatibility, memory efficiency, and vLLM integration.

**Mechanism:** `PatchFastRL()` is the main entry point that calls `_patch_trl_rl_trainers()` to dynamically patch all TRL trainer classes. Uses `RLTrainer_replacement` string template to generate modified trainer source code with injected patches from `rl_replacements.py`. `vLLMSamplingParams()` wraps vLLM's SamplingParams for GRPO generation. Patches include: gradient accumulation fixes, mixed precision handling, vLLM weight reloading removal, untrained token fixes, and custom compute_loss implementations. Handles TRL version compatibility (0.15.x through 0.26+) with conditional patching. `patch_trl_openenv()` patches experimental OpenEnv utilities for vLLM sleep/wake functionality.

**Significance:** Core component for RL training - enables GRPO and other RL methods to work efficiently with Unsloth-optimized models. Critical for the popular GRPO training workflow that combines supervised learning with reinforcement learning from rewards.
