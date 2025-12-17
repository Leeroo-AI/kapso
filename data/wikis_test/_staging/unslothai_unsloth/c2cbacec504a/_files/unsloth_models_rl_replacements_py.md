# File: `unsloth/models/rl_replacements.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 985 |
| Functions | `sft_trainer_fix_untrained_tokens`, `dpo_trainer_fix_columns`, `sft_trainer_prepare_dataset`, `sft_trainer_compute_loss`, `grpo_trainer__prepare_inputs`, `grpo_trainer__generate_single_turn`, `grpo_trainer__generate_and_score_completions`, `grpo_trainer_fix_maybe_apply_chat_template`, `... +7 more` |
| Imports | _utils, collections, device_type, importlib, inspect, os, re, textwrap, torch, unsloth_zoo |

## Understanding

**Status:** ✅ Explored

**Purpose:** RL framework function replacements registry for torch.compile optimization.

**Mechanism:** Maintains defaultdict registries of RL_FUNCTIONS, RL_EXTRA_ARGS, RL_CONFIG_CHANGES, RL_METRICS_CHANGES tracking torch-compileable function replacements across TRL components (SFT, DPO, GRPO trainers).

**Significance:** Configuration layer enabling selective function replacement for RL training optimization with specific fixes for each trainer type.
