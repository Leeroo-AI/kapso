# File: `src/peft/tuners/poly/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 103 |
| Classes | `PolyConfig` |
| Imports | __future__, dataclasses, peft, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Configuration dataclass for Poly (Polytropon) multi-task tuning parameters.

**Mechanism:** Defines PolyConfig with parameters: r (rank, default 8), n_tasks (number of tasks, default 1), n_skills (LoRA modules per layer, default 4), n_splits (for Multi-Head Routing, default 1), poly_type ("poly" only currently), target_modules, and standard PEFT options. Sets peft_type to POLY in __post_init__.

**Significance:** Configures Poly's architecture for multi-task learning where each layer contains n_skills independent LoRA adaptations, and a learned router determines task-specific mixing weights. Supports Multi-Head Routing (MHR) via n_splits > 1 for more expressive routing. References papers: Polytropon (https://huggingface.co/papers/2202.13914) and Multi-Head Routing (https://huggingface.co/papers/2211.03831).
