# File: `unsloth/models/rl.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1349 |
| Functions | `vLLMSamplingParams`, `PatchRL`, `patch_functions`, `patch_trl_rl_trainers`, `patch_trl_openenv`, `PatchFastRL` |
| Imports | inspect, os, re, rl_replacements, torch, trl, typing, unsloth_zoo |

## Understanding

**Status:** ✅ Explored

**Purpose:** Reinforcement learning trainer optimization via TRL integration and torch.compile.

**Mechanism:** Exports PatchFastRL and vLLMSamplingParams, patches TRL trainers with compiled forwards, manages RL-specific sampling parameters and inference optimization.

**Significance:** Enables efficient RL training (PPO, DPO, GRPO) through compiler optimization and vLLM integration for fast generation.
