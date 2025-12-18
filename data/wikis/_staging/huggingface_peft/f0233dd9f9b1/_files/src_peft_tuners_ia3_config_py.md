# File: `src/peft/tuners/ia3/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 112 |
| Classes | `IA3Config` |
| Imports | __future__, dataclasses, peft, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines IA3Config, the configuration dataclass for IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations), which specifies which modules to adapt and how to apply learned scaling vectors to activations.

**Mechanism:** IA3Config stores key parameters: (1) target_modules - layers to adapt (supports regex, lists, or 'all-linear' wildcard); (2) exclude_modules - layers to exclude from adaptation; (3) feedforward_modules - modules treated as feedforward (where scaling is applied to inputs instead of outputs), must be a subset of target_modules; (4) fan_in_fan_out - flag for Conv1D layers (like GPT-2); (5) modules_to_save - additional trainable modules (e.g., classification heads); (6) init_ia3_weights - whether to initialize scaling vectors to ones (True, recommended) or random values (False). The __post_init__ converts all module specifications to sets, validates that feedforward_modules is a subset of target_modules when both are sets, and sets peft_type to PeftType.IA3.

**Significance:** This configuration enables IA3 (https://huggingface.co/papers/2205.05638), an ultra-parameter-efficient PEFT method that learns per-dimension scaling vectors instead of weight matrices. For attention layers (non-feedforward), scaling is applied to outputs: y = W(x) * l_v. For feedforward layers, scaling is applied to inputs: y = W(x * l_k). This requires only d parameters per adapted layer (one scalar per dimension) versus 2*d*r for LoRA, making IA3 extremely efficient while maintaining competitive performance on many tasks.
