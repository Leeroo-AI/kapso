# File: `src/peft/tuners/oft/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 213 |
| Classes | `OFTConfig` |
| Imports | __future__, dataclasses, packaging, peft, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Configuration dataclass for OFT (Orthogonal Finetuning) method

**Mechanism:** Defines OFTConfig with parameters for rank/block size, module dropout, constrained OFT (COFT) with epsilon, block sharing, Cayley-Neumann parametrization options, target modules, and layer selection; validates rank vs block size exclusivity and version compatibility

**Significance:** Core configuration controlling OFT's orthogonal transformation behavior, with options for numerical stability (Cayley-Neumann) and constrained rotations (COFT)
