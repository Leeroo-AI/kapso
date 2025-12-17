# File: `src/peft/tuners/road/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 163 |
| Classes | `RoadModel` |
| Functions | `_adapter_names_pre_forward_hook` |
| Imports | __future__, contextlib, functools, layer, operator, peft, torch |

## Understanding

**Status:** ✅ Documented

**Purpose:** Main model class that applies RoAd adapters to Linear layers, with support for mixed batch inference where different samples use different adapters.

**Mechanism:** Extends BaseTuner with TRANSFORMERS_MODELS_TO_ROAD_TARGET_MODULES_MAPPING for defaults. The _create_and_replace method wraps Linear layers with RoadLayer variants (standard, 8-bit, 4-bit) using dispatcher pattern. The _enable_peft_forward_hooks context manager enables mixed batch inference by injecting adapter_names into forward calls via pre-hooks, validating adapter existence, and ensuring model is in eval mode.

**Significance:** High-level interface for RoAd adapters with unique mixed batch inference capability allowing different adapters per sample in the same forward pass. This enables serving multiple fine-tuned models efficiently. The dispatcher pattern cleanly handles quantization variants. Based on https://huggingface.co/papers/2409.00119.
