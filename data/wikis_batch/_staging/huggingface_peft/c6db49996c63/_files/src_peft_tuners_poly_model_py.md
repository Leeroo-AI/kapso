# File: `src/peft/tuners/poly/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 104 |
| Classes | `PolyModel` |
| Imports | config, contextlib, layer, peft, torch, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Main Poly model class that manages multi-task adaptation with skill routing.

**Mechanism:** PolyModel extends BaseTuner, wraps target Linear layers with Poly layers. Provides forward() and generate() methods that use _manage_pre_hooks context manager to register pre-hooks on all Poly Linear modules. Pre-hooks inject task_ids into kwargs before each layer's forward pass, enabling the router to compute task-specific mixing weights.

**Significance:** Orchestrates Poly's multi-task learning system by ensuring task_ids propagate to all adapter layers. The hook-based approach allows task_ids to be specified once at the model level rather than manually threading through every layer. Critical for multi-task scenarios where different examples in a batch may belong to different tasks, with the router determining optimal skill combinations per task.
