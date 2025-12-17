# File: `src/peft/tuners/shira/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 27 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** ✅ Documented

**Purpose:** Package initialization file that registers SHiRA (Sparse High Rank Adapter) method with PEFT and exports main components.

**Mechanism:** Imports ShiraConfig, ShiraLayer, Linear, and ShiraModel, then calls register_peft_method() to register "shira" as a valid PEFT method with prefix "shira_" and marked as mixed-compatible (is_mixed_compatible=True).

**Significance:** Entry point for SHiRA, a sparse high-rank adaptation method that trains only a sparse subset of weight elements. Unlike LoRA which uses low-rank updates, SHiRA can achieve high rank while maintaining parameter efficiency through sparsity. The mixed-compatible flag allows combining SHiRA with other adapter types.
