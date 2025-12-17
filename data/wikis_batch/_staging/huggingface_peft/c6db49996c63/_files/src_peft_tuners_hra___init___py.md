# File: `src/peft/tuners/hra/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 24 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** ✅ Explored

**Purpose:** Registers HRA (Householder Reflection Adaptation) as a PEFT method and exposes its public API.

**Mechanism:** Imports HRA components (HRAConfig, HRALayer, HRALinear, HRAConv2d, HRAModel) and registers the method with PEFT's method registry using register_peft_method, enabling orthogonal transformation-based fine-tuning.

**Significance:** Entry point for HRA, a parameter-efficient method using Householder reflections to learn orthogonal transformations of weight matrices. Supports both Linear and Conv2d layers with optional Gram-Schmidt orthogonalization.
