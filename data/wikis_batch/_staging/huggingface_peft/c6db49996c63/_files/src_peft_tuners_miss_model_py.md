# File: `src/peft/tuners/miss/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 130 |
| Classes | `MissModel` |
| Imports | layer, peft, torch |

## Understanding

**Status:** ✅ Documented

**Purpose:** Main model class that applies MiSS adapters to a pretrained model by wrapping Linear layers with MissLinear implementations.

**Mechanism:** Extends BaseTuner with TRANSFORMERS_MODELS_TO_MISS_TARGET_MODULES_MAPPING for default target modules. The _create_and_replace method wraps base Linear layers with MissLinear, passing r, mini_r, miss_dropout, and init_weights parameters. Only supports torch.nn.Linear layers; raises ValueError for other layer types.

**Significance:** High-level interface for applying MiSS adapters. Unlike some other PEFT methods, MiSS is specifically designed for Linear layers and does not support convolutional operations. Provides standard PEFT integration for the Householder reflection adaptation method described in https://huggingface.co/papers/2409.15371.
