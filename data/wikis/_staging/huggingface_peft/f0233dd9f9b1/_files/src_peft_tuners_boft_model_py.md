# File: `src/peft/tuners/boft/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 131 |
| Classes | `BOFTModel` |
| Imports | layer, peft, torch, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** BOFT model tuner

**Mechanism:** Extends BaseTuner to create BOFT-adapted models. Implements _create_and_replace() to inject BOFTLayer (Linear/Conv2d) with block_size, block_num, butterfly_factor, and dropout parameters. Uses _create_new_module() factory for layer instantiation.

**Significance:** Core tuner implementing BOFT/OFT methods from papers (https://huggingface.co/papers/2311.06243, https://huggingface.co/papers/2306.07280). Enables orthogonal fine-tuning for Linear and Conv2d layers.
