# File: `src/peft/tuners/lycoris_utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 263 |
| Classes | `LycorisConfig`, `LycorisLayer`, `LycorisTuner` |
| Imports | __future__, abc, dataclasses, peft, torch, tuners_utils, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Base classes and utilities for LyCORIS-family adapter methods (LoHa, LoKr, etc.).

**Mechanism:** Defines LycorisConfig (with rank/alpha patterns), LycorisLayer (abstract layer with rank/scaling/dropout management, merge/unmerge operations), and LycorisTuner (base tuner with module creation logic). Provides shared infrastructure for adapters using factorized weight decompositions.

**Significance:** Specialized foundation for LyCORIS-style methods that use different matrix factorization strategies than standard LoRA, enabling code reuse across LoHa (Hadamard product), LoKr (Kronecker product), and similar decomposition-based techniques.
