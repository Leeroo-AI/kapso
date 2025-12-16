# File: `unsloth/models/rl_replacements.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 985 |
| Functions | `sft_trainer_fix_untrained_tokens`, `dpo_trainer_fix_columns`, `sft_trainer_prepare_dataset`, `sft_trainer_compute_loss`, `grpo_trainer__prepare_inputs`, `grpo_trainer__generate_single_turn`, `grpo_trainer__generate_and_score_completions`, `grpo_trainer_fix_maybe_apply_chat_template`, `... +7 more` |
| Imports | _utils, collections, device_type, importlib, inspect, os, re, textwrap, torch, unsloth_zoo |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides specific function-level replacements and enhancements for TRL trainers, containing the actual optimization logic that `rl.py` injects into trainer classes.

**Mechanism:**
- **Dictionary-based registration system** using `defaultdict(list)`:
  - `RL_EXTRA_ARGS`: Additional initialization arguments per trainer
  - `RL_FUNCTIONS`: Function replacements per trainer
  - `RL_PRE_ITEMS`: Code to inject before trainer definition
  - `RL_CONFIG_CHANGES`: Config class modifications
  - `RL_METRICS_CHANGES`: Metrics tracking additions
  - `RL_ADDITIONAL_FUNCTIONS`: Additional patching functions

- **SFT (Supervised Fine-Tuning) optimizations**:
  - `sft_trainer_fix_untrained_tokens` (lines 61-72): Detects and fixes untrained vocabulary tokens
  - `sft_trainer_prepare_dataset` (lines 99-170): Fixes double BOS token issues by detecting when BOS is in prompt/template and disabling `add_special_tokens`
  - `sft_trainer_compute_loss` (lines 178-194): Simplified loss computation override

- **DPO (Direct Preference Optimization) fixes**:
  - `dpo_trainer_fix_columns` (lines 79-92): Removes redundant columns that cause tokenization issues

- **GRPO (Group Relative Policy Optimization) optimizations** (most extensive):
  - `grpo_trainer__prepare_inputs` (lines 201-217): Adds mixed precision autocasting for forward passes
  - `grpo_trainer__generate_single_turn` (lines 227-239): Removes unnecessary weight reloading
  - `grpo_trainer__generate_and_score_completions` (lines 246-391): Major optimizations:
    - Fixes special token handling (skip_special_tokens should be False)
    - Left-pads prompts before computing hidden states for proper alignment
    - Adds vLLM sleep/wake_up calls for memory efficiency
    - Handles max_prompt_length truncation with protected tokens
    - Fixes sampling_per_token_logps collection
  - `grpo_trainer__get_per_token_logps` (lines 465-512): Returns None for Unsloth's efficient GRPO path, avoiding redundant logit computation
  - `grpo_trainer__get_per_token_logps_and_entropies` (lines 518-618): Computes logits with mixed precision and optional entropy
  - `grpo_trainer_compute_loss` (lines 639-876): Core GRPO loss computation:
    - Uses chunked computation via `grpo_accumulated_loss` for memory efficiency
    - Handles importance sampling corrections
    - Supports multiple loss types (GRPO, Dr GRPO, DAPO, BNPO)
    - Collects extensive metrics (KL divergence, completion length, sampling deltas, importance ratios)
  - `grpo_trainer_fix_batch_size` (lines 884-898): Ensures batch size is multiple of num_generations
  - `grpo_trainer_metrics` (lines 905-932): Registers reward function metrics
  - `grpo_trainer_fix_maybe_apply_chat_template` (lines 398-443): Fixes chat template application to support extra kwargs like `reasoning_effort`

- **OpenEnv patching**:
  - `openenv_vllm_reload_weights` (lines 937-982): Patches TRL's OpenEnv to remove weight reloading and fix wake_up calls

**Significance:** This file contains the **actual optimization implementations** that make Unsloth-accelerated RL training work. The GRPO optimizations are particularly sophisticated, implementing memory-efficient chunked computation, proper mixed precision handling, and extensive metric tracking. The fixes address numerous edge cases and bugs in TRL's implementations (double BOS tokens, incorrect special token handling, weight reloading overhead). Together with `rl.py`, this forms Unsloth's complete RL training stack, enabling state-of-the-art algorithms like GRPO to run efficiently on consumer GPUs. The chunking strategy (`unsloth_num_chunks`) is especially important for fitting large models in limited VRAM during RL training.
