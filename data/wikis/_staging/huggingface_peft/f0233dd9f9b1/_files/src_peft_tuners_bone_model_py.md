# File: `src/peft/tuners/bone/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 126 |
| Classes | `BoneModel` |
| Imports | layer, peft, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Bone model tuner

**Mechanism:** Extends BaseTuner to create Bone-adapted models. Implements _create_and_replace() to inject BoneLinear layers with rank r and init_weights parameters. Uses _create_new_module() factory supporting only torch.nn.Linear base layers.

**Significance:** Core tuner implementing Householder reflection adaptation from paper (https://huggingface.co/papers/2409.15371). Supports Stable Diffusion and other architectures. Limited to Linear layer adaptation.
