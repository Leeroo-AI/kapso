# File: `src/peft/tuners/oft/layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 950 |
| Classes | `MultiplicativeDropoutLayer`, `OFTRotationModule`, `OFTLayer`, `Linear`, `Conv2d` |
| Functions | `dispatch_default` |
| Imports | __future__, config, peft, torch, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implementation of OFT layer adaptations for Linear and Conv2d layers

**Mechanism:** Uses OFTRotationModule to create orthogonal transformations via Cayley parametrization or Cayley-Neumann approximation on skew-symmetric matrices; constructs block diagonal orthogonal matrices; applies to inputs before base layer computation; supports COFT constraint projection and multiplicative dropout

**Significance:** Core implementation of OFT's orthogonal rotation transformations enabling parameter-efficient finetuning with orthogonality preservation and optional numerical optimizations
