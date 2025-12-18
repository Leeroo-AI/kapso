# File: `src/peft/tuners/c3a/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 101 |
| Classes | `C3AModel` |
| Imports | __future__, itertools, layer, peft, re, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** C3A model tuner

**Mechanism:** Extends BaseTuner to create C3A-adapted models. Implements _create_and_replace() with regex pattern matching for block_size_pattern lookup, injects C3ALinear layers with block_size and init_weights. Uses _create_new_module() factory supporting only torch.nn.Linear base layers.

**Significance:** Core tuner implementing circular convolution adapter from paper (https://huggingface.co/papers/2407.19342). Pattern-based block_size selection enables fine-grained per-layer control. Limited to Linear layer adaptation.
