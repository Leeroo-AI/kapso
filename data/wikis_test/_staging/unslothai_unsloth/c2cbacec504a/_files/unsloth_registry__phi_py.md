# File: `unsloth/registry/_phi.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 74 |
| Classes | `PhiModelInfo` |
| Functions | `register_phi_4_models`, `register_phi_4_instruct_models`, `register_phi_models` |
| Imports | unsloth |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Registers Microsoft Phi 4 model variants in base and instruction-tuned versions.

**Mechanism:** Defines PhiModelInfo class with simple version-based naming (phi-4), creates two ModelMeta configurations (base and mini-instruct) for 1B size only, supports multiple quantization types.

**Significance:** Integrates Microsoft's lightweight Phi-4 model family with instruction-tuning variants, targeting efficient inference on resource-constrained systems.
