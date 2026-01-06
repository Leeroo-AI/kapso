# File: `src/peft/tuners/gralora/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 24 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package initialization file for the GraLoRA (Granular Low-Rank Adaptation) PEFT method that exports key components and registers GraLoRA as a PEFT method in the library.

**Mechanism:** The file imports GraloraConfig, GraloraLayer, and GraloraModel from their respective modules, exposes them via __all__, and calls register_peft_method() to register GraLoRA with the PEFT framework, mapping the name "gralora" to GraloraConfig and GraloraModel. This registration enables users to instantiate GraLoRA adapters using the standard PEFT API.

**Significance:** This is a core initialization file that makes GraLoRA available as a first-class PEFT method. GraLoRA is an advanced low-rank adaptation technique that divides weight matrices into subblocks and applies block-wise low-rank updates with information exchange between blocks, achieving higher expressivity than standard LoRA with the same parameter count. The registration here integrates it into PEFT's adapter ecosystem.
