# File: `src/peft/mapping_func.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 128 |
| Functions | `get_peft_model` |
| Imports | __future__, auto, config, mapping, mixed_model, peft_model, transformers, tuners, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Main factory function `get_peft_model()` for creating PEFT models.

**Mechanism:** Takes a pretrained model and PeftConfig, returns wrapped PeftModel. Handles: updating `base_model_name_or_path` in config, warning if model already has PEFT layers, revision tracking. If `mixed=True`, returns `PeftMixedModel` for mixing adapter types. Otherwise selects appropriate PeftModel subclass based on `task_type` via `MODEL_TYPE_TO_PEFT_MODEL_MAPPING`. Passes through `autocast_adapter_dtype` and `low_cpu_mem_usage` options. Warns about adapter name conflicts with PEFT prefixes.

**Significance:** Primary entry point for creating PEFT models. Most user code starts with `model = get_peft_model(base_model, config)`.
