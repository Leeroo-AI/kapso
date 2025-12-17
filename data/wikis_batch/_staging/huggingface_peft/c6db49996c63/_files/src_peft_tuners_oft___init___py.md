# File: `src/peft/tuners/oft/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 52 |
| Imports | config, gptq, layer, model, peft |

## Understanding

**Status:** ✅ Explored

**Purpose:** Module initialization and registration for OFT (Orthogonal Finetuning) method

**Mechanism:** Imports and exposes OFT classes (OFTConfig, OFTLayer, OFTModel, Linear, Conv2d, GPTQOFTLinear), registers "oft" as PEFT method, uses lazy imports via __getattr__ for optional quantization backends (bnb 8bit/4bit, eetq)

**Significance:** Entry point for OFT tuner with flexible quantization support, enabling orthogonal transformations for parameter-efficient finetuning
