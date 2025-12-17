# File: `src/transformers/trainer_jit_checkpoint.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 126 |
| Classes | `CheckpointManager`, `JITCheckpointCallback` |
| Imports | os, signal, threading, trainer_callback, trainer_utils, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements Just-In-Time (JIT) checkpointing that saves model state when receiving SIGTERM signals, enabling graceful termination in preemptible compute environments like shared clusters with Kubernetes or Slurm.

**Mechanism:** CheckpointManager registers a SIGTERM signal handler that sets is_checkpoint_requested flag after a configurable grace period (default 3 seconds) to distinguish SIGTERM from SIGKILL. JITCheckpointCallback extends TrainerCallback and monitors this flag during training events (on_step_begin, on_step_end, on_pre_optimizer_step, on_epoch_end). When triggered, it executes execute_jit_checkpoint() which creates a checkpoint directory, writes a sentinel file to indicate in-progress checkpointing, invokes trainer._save_checkpoint(), removes the sentinel on success, and sets control.should_training_stop to exit gracefully. The callback restores the original SIGTERM handler on training completion.

**Significance:** Critical for production training on preemptible infrastructure where jobs can be terminated with minimal notice. Prevents loss of training progress by ensuring checkpoints are saved before termination. Particularly valuable for long-running training jobs on spot instances, Kubernetes with Kueue, or Slurm with time limits, where recovery from interruptions is essential.
