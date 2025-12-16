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

**Purpose:** Registers Microsoft's Phi-4 model family (base and instruction-tuned variants) with simple version-based naming.

**Mechanism:** Defines `PhiModelInfo` class that constructs names as "phi-{version}" (e.g., "phi-4"). Creates two ModelMeta instances: `PhiMeta4` for base models with no instruct tag, and `PhiInstructMeta4` for instruction-tuned models with "mini-instruct" tag. Both specify org="microsoft", version="4", size="1" (presumably 1B parameters), and are text-only models. Base models support NONE/BNB/UNSLOTH quantization, while instruct models add GGUF support. Registration functions use global flags for one-time registration.

**Significance:** Enables access to Microsoft's Phi-4 models, which are small-scale efficient models designed for resource-constrained environments. The "mini-instruct" tag distinguishes instruction-following variants. Phi models represent Microsoft's contribution to compact, efficient LLMs suitable for edge deployment or low-resource scenarios.
