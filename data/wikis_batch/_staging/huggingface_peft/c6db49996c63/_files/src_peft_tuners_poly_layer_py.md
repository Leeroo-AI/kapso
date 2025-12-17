# File: `src/peft/tuners/poly/layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 165 |
| Classes | `PolyLayer`, `Linear` |
| Imports | config, math, peft, router, torch, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements Poly adapter layers with multiple LoRA skills and task-based routing.

**Mechanism:** PolyLayer stores poly_lora_A (n_splits × n_skills × in_features//n_splits × r) and poly_lora_B (n_splits × n_skills × r × out_features//n_splits) parameters, plus poly_router module. Forward pass gets mixing_weights (bs × n_splits × n_skills) from router based on task_ids, combines LoRAs via einsum: A = einsum("bqs,qsdr->bqdr", mixing_weights, poly_lora_A), reshapes to batch matrices, and applies result += x.bmm(A).bmm(B) / r. A initialized with Kaiming, B initialized to zeros (if init_weights=True).

**Significance:** Implements Poly's multi-skill architecture where each layer has n_skills LoRA modules that are dynamically combined per task. The router learns optimal skill composition for each task, enabling parameter sharing across tasks. The batch matrix multiplication efficiently applies different LoRA combinations to different examples in a batch based on their task_ids.
