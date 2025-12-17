# File: `src/peft/tuners/hra/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 131 |
| Classes | `HRAModel` |
| Imports | layer, peft, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Orchestrates HRA model adaptation by replacing target layers with Householder reflection-enhanced versions.

**Mechanism:** HRAModel extends BaseTuner, implementing _create_and_replace to substitute nn.Linear with HRALinear and nn.Conv2d with HRAConv2d. Manages r (reflection count) and apply_GS (Gram-Schmidt flag) parameters. Uses TRANSFORMERS_MODELS_TO_HRA_TARGET_MODULES_MAPPING for architecture-specific defaults. Provides example usage for Stable Diffusion (text encoder and UNet adaptation).

**Significance:** High-level adapter enabling orthogonal transformation-based fine-tuning via Householder reflections. Paper: https://huggingface.co/papers/2405.17484. Particularly suited for diffusion models and vision tasks. Orthogonal transformations preserve gradient norms and avoid catastrophic changes to pretrained representations.
