# File: `src/peft/tuners/loha/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 24 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** ✅ Documented

**Purpose:** Package initialization file that registers the LoHa (Low-Rank Hadamard Product) adapter method with PEFT and exports the main components for external use.

**Mechanism:** Imports the LoHaConfig, LoHaLayer implementations (Conv2d, Linear), and LoHaModel classes, then calls register_peft_method() to register "loha" as a valid PEFT method with prefix "hada_" and marks it as compatible with mixed adapter configurations.

**Significance:** This is the entry point for the LoHa tuning method. It enables users to use LoHa adapters by making the method discoverable within the PEFT framework. The registration ensures proper integration with PEFT's adapter management system.
