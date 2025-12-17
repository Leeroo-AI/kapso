# File: `src/peft/tuners/bone/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 129 |
| Classes | `BoneConfig` |
| Imports | __future__, dataclasses, peft, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Configuration dataclass for BONE (Block Affine) adaptation method

**Mechanism:** Defines BoneConfig with rank parameter, target modules, initialization options (bone/bat variants), layer selection; warns that BONE will be removed in v0.19.0 in favor of MISS

**Significance:** Configuration for block-wise affine transformations, supporting two variants (BONE and BAT structures) via different initialization methods
