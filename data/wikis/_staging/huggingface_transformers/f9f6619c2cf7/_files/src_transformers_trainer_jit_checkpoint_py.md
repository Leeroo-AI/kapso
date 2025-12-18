# File: `src/transformers/trainer_jit_checkpoint.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 126 |
| Classes | `CheckpointManager`, `JITCheckpointCallback` |
| Imports | os, signal, threading, trainer_callback, trainer_utils, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements Just-In-Time (JIT) checkpointing for graceful training interruption on cloud platforms. Saves checkpoints when receiving SIGTERM signals (indicating impending preemption or spot instance termination).

**Mechanism:** `CheckpointManager` registers a SIGTERM signal handler that sets `is_checkpoint_requested=True` after a grace period (default 3s, to distinguish SIGTERM from SIGKILL). `JITCheckpointCallback` checks this flag at multiple points in the training loop (`on_pre_optimizer_step`, `on_step_begin`, `on_step_end`, `on_epoch_end`). When triggered, it calls `execute_jit_checkpoint()` which creates a checkpoint directory, writes a sentinel file to mark incomplete checkpoints, calls `trainer._save_checkpoint()`, removes the sentinel on success, and sets `control.should_training_stop=True` to exit gracefully.

**Significance:** Critical for reliable training on cloud platforms with spot instances or preemptible VMs (AWS Spot, GCP Preemptible, Azure Spot). Prevents data loss from sudden terminations by saving checkpoints before shutdown. Enabled via `TrainingArguments.enable_jit_checkpoint`. The sentinel file mechanism helps identify corrupted checkpoints from incomplete saves. Particularly valuable for long-running training jobs where checkpoint frequency is low to minimize I/O overhead.
