# File: `src/peft/tuners/ia3/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 112 |
| Classes | `IA3Config` |
| Imports | __future__, dataclasses, peft, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Configuration class for IA3 method with feedforward module distinction

**Mechanism:** IA3Config stores target_modules (where to apply IA3), exclude_modules, feedforward_modules (special handling - scale inputs not outputs), fan_in_fan_out, modules_to_save, init_ia3_weights. Validates feedforward_modules is subset of target_modules, converts lists to sets for efficient lookup

**Significance:** Distinguishes feedforward vs attention treatment - critical for IA3 method which scales inputs for feedforward layers but outputs for attention, following the original paper's asymmetric scaling approach
