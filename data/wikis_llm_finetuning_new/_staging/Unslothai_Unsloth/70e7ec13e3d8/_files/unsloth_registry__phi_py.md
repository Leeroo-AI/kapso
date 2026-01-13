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

**Purpose:** Defines model metadata and registration logic for Microsoft's Phi-4 model family (base and mini-instruct variants).

**Mechanism:** Provides `PhiModelInfo` class that constructs model names in the format `phi-{version}`. Defines two `ModelMeta` configurations: `PhiMeta4` for base Phi-4 and `PhiInstructMeta4` for the mini-instruct variant. Both currently support only the 1B size. The instruct variant additionally supports GGUF quantization. Uses the "mini-instruct" tag to match Microsoft's naming convention.

**Significance:** Enables Unsloth support for Phi-4 models. The simpler naming scheme (version-only without size in the key) reflects Phi's naming convention. Like other model files, includes `__main__` verification for Hub existence validation.
