# File: `src/peft/tuners/gralora/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 24 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** ✅ Explored

**Purpose:** Registers GraLoRA (Gradient Low-Rank Adaptation) as a PEFT method and exposes its public API.

**Mechanism:** Imports GraLoRA components (GraloraConfig, GraloraLayer, GraloraModel) and registers the method with PEFT's method registry using register_peft_method, enabling block-wise low-rank adaptation with information exchange.

**Significance:** Entry point for GraLoRA, a variant of LoRA that uses block-structured low-rank decomposition with gralora_k subblocks to increase expressivity without increasing parameter count. Supports hybrid mode combining GraLoRA and vanilla LoRA.
