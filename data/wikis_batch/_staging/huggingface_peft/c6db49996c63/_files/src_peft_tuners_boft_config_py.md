# File: `src/peft/tuners/boft/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 160 |
| Classes | `BOFTConfig` |
| Imports | __future__, dataclasses, peft, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Configuration dataclass for BOFT (Butterfly Orthogonal Finetuning via Butterfly Factorization)

**Mechanism:** Defines BOFTConfig with parameters for block size, block number, butterfly factors, dropout, target modules, and layer selection; validates that block size and block number constraints are met

**Significance:** Core configuration component that controls BOFT's behavior, based on ICLR 2024 paper using butterfly factorization for parameter-efficient orthogonal finetuning
