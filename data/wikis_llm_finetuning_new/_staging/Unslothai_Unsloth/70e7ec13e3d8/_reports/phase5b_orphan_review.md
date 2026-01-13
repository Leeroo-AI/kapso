# Phase 6b: Orphan Review Report

## Summary
- MANUAL_REVIEW files evaluated: 13
- Approved: 4
- Rejected: 9

## Decisions

| File | Decision | Reasoning |
|------|----------|-----------|
| `unsloth/_auto_install.py` | REJECTED | Install helper script, no public API |
| `unsloth/dataprep/synthetic_configs.py` | REJECTED | Config template string, no algorithm |
| `unsloth/device_type.py` | APPROVED | Public API, user-facing device detection |
| `unsloth/models/dpo.py` | REJECTED | Stub functions, no implementation |
| `unsloth/registry/_deepseek.py` | APPROVED | Public registration API, model metadata |
| `unsloth/registry/_gemma.py` | REJECTED | Internal registry helper, small |
| `unsloth/registry/_llama.py` | REJECTED | Internal registry helper, small |
| `unsloth/registry/_mistral.py` | REJECTED | Internal registry helper, small |
| `unsloth/registry/_phi.py` | REJECTED | Internal registry helper, small |
| `unsloth/registry/_qwen.py` | REJECTED | Internal registry helper, small |
| `unsloth/registry/registry.py` | APPROVED | Core public API for model registry |
| `unsloth/utils/attention_dispatch.py` | APPROVED | Public API, attention algorithm |
| `unsloth/utils/hf_hub.py` | REJECTED | Thin wrapper, no distinct algorithm |

## Notes

### Patterns Observed
- Most registry files with underscore prefix (`_deepseek.py`, `_gemma.py`, etc.) are internal helpers that register model metadata. Only `_deepseek.py` was large enough (206 lines) with sufficient complexity to warrant documentation.
- The core `registry.py` file contains the foundational `ModelInfo`, `ModelMeta`, and `QuantType` classes that power the entire registry system.
- `attention_dispatch.py` implements actual attention backend selection logic with FlashAttention, xFormers, and SDPA backends.

### Borderline Cases
- `_deepseek.py` was borderline due to underscore prefix, but approved because:
  - Largest of the model registry files (206 lines)
  - Contains multiple model classes (`DeepseekV3ModelInfo`, `DeepseekR1ModelInfo`)
  - Demonstrates the pattern for how models are registered

- `_qwen.py` (136 lines) was borderline but rejected because:
  - Still uses internal underscore naming
  - Follows same pattern as other smaller registry files
  - Not substantially different from `_gemma.py` or `_phi.py`

### Evaluation Criteria Applied
1. **Public API check**: Files with `__all__` exports or public classes/functions
2. **User-facing**: Would users import/call this directly?
3. **Distinct algorithm**: Does it implement something beyond configuration/glue code?
