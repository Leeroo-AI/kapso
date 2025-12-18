# File: `src/peft/tuners/hra/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 131 |
| Classes | `HRAModel` |
| Imports | layer, peft, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements HRAModel, the main adapter model class that applies HRA (Householder Reflection Adaptation) to pretrained models by replacing target Linear and Conv2d layers with HRA-enabled versions.

**Mechanism:** HRAModel extends BaseTuner and implements _create_and_replace() to swap target layers with HRA versions. The method extracts adapter parameters (r, apply_GS, init_weights) from the config and passes them to either update_layer() for existing HRA layers or _create_new_module() for new replacements. The _create_new_module() validates layer types (torch.nn.Linear or torch.nn.Conv2d) and creates either HRALinear or HRAConv2d modules accordingly, raising errors for unsupported layer types. Each HRA layer stores Householder reflection vectors that define orthogonal transformations of the base weights.

**Significance:** This is the core model class for HRA (https://huggingface.co/papers/2405.17484), responsible for converting pretrained models into HRA-adapted versions using Householder reflections. Unlike low-rank methods, HRA preserves full rank while applying orthogonal transformations. The target module mapping uses TRANSFORMERS_MODELS_TO_HRA_TARGET_MODULES_MAPPING for architecture-specific defaults. HRA is particularly effective for vision tasks (Stable Diffusion text encoders and UNets) and supports both Linear and Conv2d layers, making it more versatile than many other PEFT methods.
