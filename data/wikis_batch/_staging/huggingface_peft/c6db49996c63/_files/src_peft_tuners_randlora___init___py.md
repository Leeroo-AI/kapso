# File: `src/peft/tuners/randlora/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 40 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** ✅ Documented

**Purpose:** Package initialization file that registers RandLoRA method with PEFT and provides lazy imports for quantized layer implementations (Linear8bitLt, Linear4bit).

**Mechanism:** Imports RandLoraConfig, RandLoraLayer, Linear, and RandLoraModel, then calls register_peft_method() to register "randlora" with prefix "randlora_". Uses __getattr__ for conditional lazy importing of bnb quantized layers only when bitsandbytes is available and the specific classes are accessed.

**Significance:** Entry point for RandLoRA (Random Low-Rank Adaptation) from https://huggingface.co/papers/2502.00987. The lazy import pattern avoids hard dependencies on bitsandbytes while supporting quantized models when available.
