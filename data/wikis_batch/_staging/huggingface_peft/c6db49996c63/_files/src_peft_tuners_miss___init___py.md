# File: `src/peft/tuners/miss/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 24 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** ✅ Documented

**Purpose:** Package initialization file that registers the MiSS (Mixture of Sharded Squares) adapter method with PEFT and exports main components.

**Mechanism:** Imports MissConfig, MissLayer, MissLinear, and MissModel classes, then calls register_peft_method() to register "miss" as a valid PEFT method. Unlike some other methods, this does not specify a prefix or mixed_compatible flag.

**Significance:** Entry point for the MiSS tuning method. MiSS is based on Householder reflection adaptation for efficient fine-tuning. The registration makes MiSS discoverable and usable through PEFT's standard interfaces.
