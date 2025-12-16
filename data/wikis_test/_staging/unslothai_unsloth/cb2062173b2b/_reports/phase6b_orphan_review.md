# Phase 6b: Orphan Review Report

## Summary
- MANUAL_REVIEW files evaluated: 13
- Approved: 4
- Rejected: 9

## Decisions
| File | Decision | Reasoning |
|------|----------|-----------|
| `unsloth/_auto_install.py` | REJECTED | Internal install script, no public API |
| `unsloth/dataprep/synthetic_configs.py` | REJECTED | Just config string template, no distinct code |
| `unsloth/device_type.py` | APPROVED | Public API, user-facing device detection |
| `unsloth/models/dpo.py` | REJECTED | Empty stub functions, no implementation |
| `unsloth/models/qwen2.py` | APPROVED | Public model class, user-facing API |
| `unsloth/registry/_deepseek.py` | REJECTED | Private module, internal registry data |
| `unsloth/registry/_gemma.py` | REJECTED | Private module, internal registry data |
| `unsloth/registry/_llama.py` | REJECTED | Private module, internal registry data |
| `unsloth/registry/_mistral.py` | REJECTED | Private module, internal registry data |
| `unsloth/registry/_phi.py` | REJECTED | Private module, internal registry data |
| `unsloth/registry/_qwen.py` | REJECTED | Private module, internal registry data |
| `unsloth/registry/registry.py` | APPROVED | Public API, core registry system |
| `unsloth/utils/attention_dispatch.py` | APPROVED | Public API, implements attention dispatch |

## Notes

### Patterns Observed
- All registry files prefixed with `_` (e.g., `_deepseek.py`, `_gemma.py`) contain model metadata configurations that are internal to the registry system. They define model variants and quantization types but are not intended for direct user import.
- The core `registry.py` file provides the public API (`ModelInfo`, `ModelMeta`, `QuantType`, `MODEL_REGISTRY`) that users would interact with for custom model registration.

### Borderline Files
- **`unsloth/models/qwen2.py`** (101 lines): Smaller file but contains `FastQwen2Model` class with `from_pretrained` method - this is user-facing model loading API.
- **`unsloth/device_type.py`** (98 lines): Smaller file but exports critical utilities (`get_device_type`, `DEVICE_TYPE`, `is_hip`) that users may need for hardware detection.

### Key Approvals Rationale
1. **`device_type.py`**: Has explicit `__all__` export list with 7 public symbols. Essential for multi-GPU/multi-platform support.
2. **`qwen2.py`**: Contains `FastQwen2Model` class that mirrors the pattern of other model adapters in AUTO_KEEP.
3. **`registry.py`**: Core data structures for the model registry system. Would be needed by anyone extending the registry.
4. **`attention_dispatch.py`**: Has explicit `__all__` with 4 public symbols. Implements attention backend selection algorithm supporting Flash Attention, xFormers, and SDPA.
