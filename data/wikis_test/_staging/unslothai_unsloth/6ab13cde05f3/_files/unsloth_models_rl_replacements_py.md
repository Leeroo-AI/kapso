# File: `unsloth/models/rl_replacements.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 985 |
| Functions | `sft_trainer_fix_untrained_tokens`, `dpo_trainer_fix_columns`, `sft_trainer_prepare_dataset`, `sft_trainer_compute_loss`, `grpo_trainer__prepare_inputs`, `grpo_trainer__generate_single_turn`, `grpo_trainer__generate_and_score_completions`, `grpo_trainer_fix_maybe_apply_chat_template`, `... +7 more` |
| Imports | _utils, collections, device_type, importlib, inspect, os, re, textwrap, torch, unsloth_zoo |

## Understanding

**Status:** ✅ Explored

**Purpose:** RL trainer function replacements

**Mechanism:** Collected function replacements for SFT/DPO/GRPO trainers

**Significance:** Centralizes RL trainer modifications
