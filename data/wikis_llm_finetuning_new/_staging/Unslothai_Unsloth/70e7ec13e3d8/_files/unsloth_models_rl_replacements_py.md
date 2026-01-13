# File: `unsloth/models/rl_replacements.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 995 |
| Functions | `sft_trainer_fix_untrained_tokens`, `dpo_trainer_fix_columns`, `sft_trainer_prepare_dataset`, `sft_trainer_compute_loss`, `grpo_trainer__prepare_inputs`, `grpo_trainer__generate_single_turn`, `grpo_trainer__generate_and_score_completions`, `grpo_trainer_fix_maybe_apply_chat_template`, `... +7 more` |
| Imports | _utils, collections, device_type, importlib, inspect, os, re, textwrap, torch, unsloth_zoo |

## Understanding

**Status:** Explored

**Purpose:** Defines specific function replacements and patches for TRL trainers (SFT, DPO, GRPO), organized into dictionaries that the main `rl.py` patching system consumes.

**Mechanism:** Exports dictionaries: `RL_EXTRA_ARGS`, `RL_FUNCTIONS`, `RL_PRE_ITEMS`, `RL_CONFIG_CHANGES`, `RL_METRICS_CHANGES` keyed by trainer name. Each dictionary contains lists of patch functions. SFT patches: `sft_trainer_fix_untrained_tokens` (fixes zero-training-loss tokens), `sft_trainer_prepare_dataset` (handles double BOS tokens). DPO patches: `dpo_trainer_fix_columns` (removes pre-tokenized columns). GRPO patches: `grpo_trainer__prepare_inputs` (adds mixed precision autocast), `grpo_trainer__get_per_token_logps` (efficient logprob computation), `grpo_trainer_compute_loss` (custom loss with KL divergence, importance sampling), `grpo_trainer__generate_and_score_completions` (vLLM sleep/wake integration, left padding fixes). Imports optimized implementations from `unsloth_zoo.rl_replacements` including `grpo_compute_loss`, `UnslothEfficientGRPO`, `grpo_accumulated_loss`.

**Significance:** Core RL implementation details - contains the actual patch logic that makes RL training work correctly and efficiently. The GRPO patches are particularly important, enabling memory-efficient policy gradient training with proper handling of vLLM generation, importance sampling, and gradient accumulation.
