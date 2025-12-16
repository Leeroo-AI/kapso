# File: `unsloth/models/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 30 |
| Imports | _utils, dpo, granite, llama, loader, mistral, qwen2, qwen3, qwen3_moe, rl |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package initialization file that exposes the public API for Unsloth's model implementations. Serves as the entry point for users to access optimized model classes for various architectures.

**Mechanism:** Imports and re-exports key components from submodules including FastLanguageModel, FastVisionModel, FastTextModel, and architecture-specific implementations (FastLlamaModel, FastMistralModel, FastQwen2Model, etc.). Also imports utility functions like is_bfloat16_supported and version information. Includes a try-except block for falcon_h1 support to handle older transformers versions gracefully.

**Significance:** Critical module entry point that defines the public interface for the models package. Users interact with this file when importing from unsloth.models to access model loaders, trainers (DPO, KTO), and reinforcement learning utilities. The selective imports ensure only supported model architectures are exposed based on the transformers version.
