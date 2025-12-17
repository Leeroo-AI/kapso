# File: `src/peft/tuners/road/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 47 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** ✅ Documented

**Purpose:** Package initialization file that registers RoAd (Rotation Adaptation) method with PEFT and provides lazy imports for quantized layer implementations.

**Mechanism:** Imports RoadConfig, RoadLayer, Linear, and RoadModel, then calls register_peft_method() to register "road" as mixed-compatible with is_mixed_compatible=True. Uses __getattr__ for conditional lazy importing of Linear8bitLt and Linear4bit quantized layers when bitsandbytes is available. Based on implementation from https://github.com/ppetrushkov/peft/tree/road (not from paper authors).

**Significance:** Entry point for RoAd method from https://huggingface.co/papers/2409.00119, which adapts models by applying learned 2D rotations to pairs of features. The mixed-compatible flag allows combining RoAd with other adapter types.
