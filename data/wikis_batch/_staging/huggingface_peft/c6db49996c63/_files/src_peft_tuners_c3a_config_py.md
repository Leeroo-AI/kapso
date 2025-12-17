# File: `src/peft/tuners/c3a/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 137 |
| Classes | `C3AConfig` |
| Imports | __future__, dataclasses, peft, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines configuration parameters for C3A (Circulant Channel-Wise Convolution) fine-tuning method.

**Mechanism:** C3AConfig dataclass extends PeftConfig with C3A-specific parameters including block_size (must divide both input and output dimensions), target_modules, bias settings, layer selection, and initialization options (xavier_uniform, kaiming_uniform, gaussian, or zeros). Validates configuration constraints in __post_init__.

**Significance:** Core configuration class controlling C3A's block circulant matrix decomposition. The block_size parameter determines parameter efficiency - larger values mean fewer parameters. Supports per-layer block size customization via block_size_pattern dictionary.
