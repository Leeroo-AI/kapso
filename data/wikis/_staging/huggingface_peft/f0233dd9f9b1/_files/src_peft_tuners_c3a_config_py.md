# File: `src/peft/tuners/c3a/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 137 |
| Classes | `C3AConfig` |
| Imports | __future__, dataclasses, peft, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** C3A configuration dataclass

**Mechanism:** Defines C3AConfig with block_size (must divide input/output dimensions), target_modules, bias type, layers_to_transform, block_size_pattern (per-layer block size overrides), and init_weights (True=zeros, False/xavier_uniform/kaiming_uniform/gaussian for initialization). Validates layers_pattern/layers_to_transform compatibility with regex target_modules.

**Significance:** Core configuration for C3A. block_size controls parameter efficiency vs. expressiveness tradeoff. block_size_pattern enables layer-specific customization. Essential for circular convolution adapter configuration.
