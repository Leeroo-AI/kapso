# Phase 6b: Orphan Review Report

## Summary
- MANUAL_REVIEW files evaluated: 11
- Approved: 2
- Rejected: 9

## Decisions

| File | Decision | Reasoning |
|------|----------|-----------|
| `unsloth/_auto_install.py` | REJECTED | Internal install helper script, no API |
| `unsloth/dataprep/synthetic_configs.py` | REJECTED | Config template string only, no API |
| `unsloth/models/dpo.py` | REJECTED | Empty stub functions, no implementation |
| `unsloth/models/qwen2.py` | APPROVED | Public API class, user-facing model loader |
| `unsloth/registry/_deepseek.py` | REJECTED | Internal registry module (underscore prefix) |
| `unsloth/registry/_gemma.py` | REJECTED | Internal registry module (underscore prefix) |
| `unsloth/registry/_llama.py` | REJECTED | Internal registry module (underscore prefix) |
| `unsloth/registry/_mistral.py` | REJECTED | Internal registry module (underscore prefix) |
| `unsloth/registry/_phi.py` | REJECTED | Internal registry module (underscore prefix) |
| `unsloth/registry/_qwen.py` | REJECTED | Internal registry module (underscore prefix) |
| `unsloth/registry/registry.py` | APPROVED | Core public API classes and functions |

## Notes

### Patterns Observed
- **Registry modules**: All `_*.py` files in the registry directory follow a consistent pattern of internal modules with underscore prefixes. They provide model metadata definitions but are not user-facing APIs. The core `registry.py` is the public API that users interact with.

- **Stub files**: `dpo.py` contains only stub functions (`PatchDPOTrainer` and `PatchKTOTrainer`) that return nothing - likely placeholders for future implementation.

- **Config files**: `synthetic_configs.py` contains only a YAML template string, no actual code logic.

### Borderline Cases
1. **`unsloth/models/qwen2.py`** (APPROVED): At only 101 lines, this was borderline. However, it contains a full `FastQwen2Model` class with `pre_patch()` and `from_pretrained()` methods that constitute a user-facing public API for loading Qwen2 models. Users would directly interact with this class.

2. **`unsloth/registry/registry.py`** (APPROVED): Contains core public classes (`QuantType`, `ModelInfo`, `ModelMeta`) and the `MODEL_REGISTRY` dictionary. This is the backbone of the model registration system and users may need to reference these types.

3. **Registry `_*.py` files** (REJECTED): While these contain substantial code (74-206 lines) and define public registration functions like `register_deepseek_models()`, the underscore prefix indicates internal use. The registration happens automatically at import time, so users don't need to call these directly.

### Summary Statistics
- Files with underscore prefix: 7 (all rejected as internal)
- Files with public classes: 2 (both approved)
- Stub/config-only files: 2 (both rejected)
