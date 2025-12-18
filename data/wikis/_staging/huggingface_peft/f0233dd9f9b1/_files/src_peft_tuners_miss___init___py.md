# File: `src/peft/tuners/miss/__init__.py`

**Category:** Core Adapter Implementation

| Property | Value |
|----------|-------|
| Lines | 24 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** Fully explored

**Purpose:** Package initialization file that exports the MiSS (Mixture of Subspaces) adapter components and registers MiSS as a PEFT method.

**Mechanism:**
1. Imports core MiSS components: `MissConfig`, `MissLayer`, `MissLinear`, `MissModel`
2. Exports all classes through `__all__` for public API access
3. Registers MiSS with PEFT framework using `register_peft_method()`:
   - Method name: "miss"
   - Configuration class: `MissConfig`
   - Model class: `MissModel`
   - No parameter prefix specified (unlike LoRA/LoHa)

**Significance:** This is the entry point for the MiSS adapter method. MiSS (introduced in 2024-2025) is a parameter-efficient fine-tuning technique based on Householder reflection adaptation. It provides an alternative to LoRA-style adapters by using matrix-free representations and subspace mixing. The registration enables MiSS as a first-class PEFT adapter type for efficient model adaptation with potentially different trade-offs than rank-based methods.

## Key Components

- **Exported Classes**: `MissConfig`, `MissLayer`, `MissLinear`, `MissModel`
- **Method Registration**: Enables MiSS as a PEFT tuning strategy
- **Paper Reference**: https://huggingface.co/papers/2409.15371
