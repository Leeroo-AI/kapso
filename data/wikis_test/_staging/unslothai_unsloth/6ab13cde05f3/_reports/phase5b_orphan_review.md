# Phase 6b: Orphan Review Report

## Summary
- MANUAL_REVIEW files evaluated: 14
- Approved: 5
- Rejected: 9

## Decisions

| File | Decision | Reasoning |
|------|----------|-----------|
| `unsloth/_auto_install.py` | REJECTED | Script prints pip cmd, no API |
| `unsloth/dataprep/synthetic_configs.py` | REJECTED | Config string only, no code |
| `unsloth/device_type.py` | APPROVED | Public API via __all__, user-facing |
| `unsloth/models/dpo.py` | REJECTED | Empty stub, no implementation |
| `unsloth/models/qwen2.py` | APPROVED | Public from_pretrained API |
| `unsloth/registry/_deepseek.py` | APPROVED | Model registry with public functions |
| `unsloth/registry/_gemma.py` | REJECTED | Internal registry, _ prefix, small |
| `unsloth/registry/_llama.py` | REJECTED | Internal registry, _ prefix |
| `unsloth/registry/_mistral.py` | REJECTED | Internal registry, _ prefix |
| `unsloth/registry/_phi.py` | REJECTED | Internal registry, _ prefix |
| `unsloth/registry/_qwen.py` | REJECTED | Internal registry, _ prefix |
| `unsloth/registry/registry.py` | APPROVED | Core API: ModelInfo, ModelMeta |
| `unsloth/utils/attention_dispatch.py` | APPROVED | Public API, backend algorithm |
| `unsloth/utils/hf_hub.py` | REJECTED | Small wrapper utility, 78 lines |

## Notes

### Patterns Observed
- Model registry files with `_` prefix follow a consistent pattern: internal metadata definitions for specific model families
- The core `registry.py` contains the public API that other registry files depend on
- Small utility files (<100 lines) that just wrap external APIs (HuggingFace Hub) were rejected

### Borderline Decisions
- **`_deepseek.py`**: Approved despite `_` prefix because it's 206 lines and defines multiple public registration functions (`register_deepseek_models`, etc.) that users may call directly
- **`_qwen.py`**: Rejected (136 lines) - similar to other model registry files but smaller and follows internal convention
- **`hf_hub.py`**: Rejected despite having public functions - it's a thin wrapper (78 lines) around HuggingFace Hub API with no distinct algorithm

### Criteria Applied
1. **Public API**: Files with `__all__` exports or public classes/functions without `_` prefix
2. **User-facing**: Would a developer import or interact with this directly?
3. **Distinct algorithm**: Does it implement substantive logic vs. glue code?
