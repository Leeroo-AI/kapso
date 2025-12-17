# File: `src/peft/tuners/lora/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 65 |
| Imports | arrow, config, eva, gptq, layer, model, peft |

## Understanding

**Status:** ✅ Documented

**Purpose:** Package initialization and public API definition for LoRA tuning module

**Mechanism:** Imports and exports core LoRA classes (LoraConfig, LoraModel), layer implementations (Linear, Embedding, Conv2d, etc.), specialized variants (Arrow, EVA), and quantization-specific adaptations (GPTQ, AWQ, AQLM, BNB, EETQ, HQQ, INC, Torchao). Acts as the central access point for all LoRA functionality.

**Significance:** Essential module entry point that exposes PEFT's complete LoRA adapter ecosystem to users. Defines what components are publicly available and provides a unified import interface for the entire LoRA subsystem.
