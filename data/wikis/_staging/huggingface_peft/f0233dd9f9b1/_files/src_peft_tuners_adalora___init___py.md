# File: `src/peft/tuners/adalora/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 43 |
| Imports | config, gptq, layer, model, peft |

## Understanding

**Status:** ✅ Explored

**Purpose:** AdaLora module initialization and registration

**Mechanism:** Exports AdaLoraConfig, AdaLoraLayer, AdaLoraModel, RankAllocator, SVDLinear, SVDQuantLinear classes and registers "adalora" as PEFT method. Uses lazy imports for BNB quantized layers (SVDLinear8bitLt, SVDLinear4bit) via __getattr__ pattern.

**Significance:** Core entry point for AdaLora (Adaptive LoRA) tuning method which dynamically allocates rank budget across layers during training. Essential for enabling adaptive parameter-efficient fine-tuning.
