# File: `unsloth/models/rl_replacements.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 995 |
| Functions | `sft_trainer_fix_untrained_tokens`, `dpo_trainer_fix_columns`, `sft_trainer_prepare_dataset`, `sft_trainer_compute_loss`, `grpo_trainer__prepare_inputs`, `grpo_trainer__generate_single_turn`, `grpo_trainer__generate_and_score_completions`, `grpo_trainer_fix_maybe_apply_chat_template`, `... +7 more` |
| Imports | _utils, collections, device_type, importlib, inspect, os, re, textwrap, torch, unsloth_zoo |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines replacement functions and code injection patterns for RL trainers (SFT, DPO, GRPO, PPO). Provides RL_EXTRA_ARGS, RL_FUNCTIONS, RL_PRE_ITEMS, RL_CONFIG_CHANGES, and RL_METRICS_CHANGES dictionaries used by rl.py to dynamically patch TRL trainers.

**Mechanism:** Exports defaultdict collections containing: 1) RL_EXTRA_ARGS: functions that generate code to inject before trainer initialization (e.g., sft_trainer_fix_untrained_tokens checks for untrained tokens, dpo_trainer_fix_columns removes DPO columns), 2) RL_FUNCTIONS: function replacement strategies (e.g., sft_trainer_prepare_dataset handles double BOS token issues, sft_trainer_compute_loss overrides mean_token_accuracy), 3) GRPO-specific patches for chat template handling, autocast precision, and generation batching. Uses regex-based code transformation to modify TRL trainer methods at import time. Imports RL_REPLACEMENTS from unsloth_zoo.rl_replacements for core replacement logic.

**Significance:** The "recipe book" for RL trainer modifications. At 995 lines, this is substantial due to the variety of RL algorithms and their quirks. The regex-based code injection is sophisticated but brittle - requires careful maintenance when TRL updates. Essential for ensuring Unsloth's optimizations work correctly with preference learning. The tokenizer double-BOS fix alone has saved countless debugging hours for users.
