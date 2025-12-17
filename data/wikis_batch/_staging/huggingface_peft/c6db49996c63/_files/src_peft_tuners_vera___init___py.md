# File: `src/peft/tuners/vera/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 40 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** ✅ Explored

**Purpose:** Module initialization and registration for the VeRA (Vector-based Random Matrix Adaptation) tuning method.

**Mechanism:** Exports core VeRA components (VeraConfig, VeraLayer, Linear, VeraModel), registers VeRA as a PEFT method with prefix "vera_lambda_", and provides lazy imports for quantized variants (Linear8bitLt, Linear4bit) when bitsandbytes is available.

**Significance:** Entry point for the VeRA tuning method, making it discoverable and usable within the PEFT framework. VeRA is a parameter-efficient alternative to LoRA that uses shared random projection matrices across layers with per-layer trainable scaling vectors.
