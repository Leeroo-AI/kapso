# File: `unsloth/models/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 30 |
| Imports | _utils, dpo, granite, llama, loader, mistral, qwen2, qwen3, qwen3_moe, rl |

## Understanding

**Status:** Explored

**Purpose:** Module entry point that exports the public API for the models subpackage, providing centralized access to all fast model classes and RL trainer patches.

**Mechanism:** Uses relative imports to gather key classes and functions from submodules (`loader.py`, `llama.py`, `dpo.py`, `rl.py`, etc.) and re-exports them. Includes conditional imports for newer model support (e.g., FalconH1 requires transformers >= 4.53.0). Exports include `FastLlamaModel`, `FastLanguageModel`, `FastVisionModel`, `FastModel`, `PatchDPOTrainer`, `PatchKTOTrainer`, `PatchFastRL`, and `vLLMSamplingParams`.

**Significance:** Core component - serves as the main public interface for the models subpackage. Users import from `unsloth.models` to access optimized model loading and RL training functionality without needing to know the internal module structure.
