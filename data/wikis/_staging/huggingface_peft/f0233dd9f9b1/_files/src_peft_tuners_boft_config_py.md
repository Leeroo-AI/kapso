# File: `src/peft/tuners/boft/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 160 |
| Classes | `BOFTConfig` |
| Imports | __future__, dataclasses, peft, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** BOFT configuration dataclass

**Mechanism:** Defines BOFTConfig with boft_block_size/boft_block_num (mutually exclusive block parameters), boft_n_butterfly_factor (butterfly decomposition depth), boft_dropout (multiplicative dropout), target_modules, bias type, and layers_to_transform. Validates that exactly one of block_size or block_num is specified.

**Significance:** Core configuration for BOFT. Block parameters control orthogonal transformation granularity. Butterfly factor determines decomposition depth for parameter efficiency vs. expressiveness tradeoff.
