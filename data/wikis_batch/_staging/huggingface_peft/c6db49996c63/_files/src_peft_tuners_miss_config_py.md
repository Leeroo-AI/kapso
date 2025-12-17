# File: `src/peft/tuners/miss/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 140 |
| Classes | `MissConfig` |
| Imports | __future__, dataclasses, peft, typing |

## Understanding

**Status:** ✅ Documented

**Purpose:** Configuration dataclass for MiSS (Mixture of Sharded Squares) adapters, supporting three variants: balance (default), bat (nonlinear updates), and mini (smaller rank with fewer parameters).

**Mechanism:** Extends PeftConfig with MiSS-specific parameters including r (rank along in_features), mini_r (rank along out_features), miss_dropout, and init_weights that accepts bool or Literal["bat", "mini"] to select variants. Validates that out_features is divisible by mini_r when using mini mode, and prevents incompatible combinations of layers_pattern/layers_to_transform with regex target_modules.

**Significance:** Provides flexible configuration for MiSS adapters with three distinct initialization strategies. The balance mode is most efficient and general, bat enables nonlinear updates across shards, and mini mode uses fewer trainable parameters. Based on the paper at https://huggingface.co/papers/2409.15371.
