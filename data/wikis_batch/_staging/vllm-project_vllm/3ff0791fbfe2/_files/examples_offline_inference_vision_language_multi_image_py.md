# File: `examples/offline_inference/vision_language_multi_image.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 1542 |
| Classes | `ModelRequestData` |
| Functions | `load_aria`, `load_aya_vision`, `load_bee`, `load_command_a_vision`, `load_deepseek_vl2`, `load_deepseek_ocr`, `load_gemma3`, `load_h2ovl`, `... +35 more` |
| Imports | PIL, argparse, dataclasses, huggingface_hub, os, transformers, typing, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Multi-image inference examples (45+ models)

**Mechanism:** Similar to vision_language.py but focused on multi-image inputs. Provides load functions for 45+ models supporting multiple images per prompt. Uses chat templates from AutoProcessor/AutoTokenizer, demonstrates proper image placeholder handling, and includes both generate() and chat() method examples. Supports MM processor caching with UUIDs.

**Significance:** Example demonstrating multi-image inference across various VLM models with correct prompt templating.
