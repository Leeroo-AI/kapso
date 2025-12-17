# File: `src/peft/tuners/shira/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 142 |
| Classes | `ShiraModel` |
| Imports | __future__, layer, peft, torch, warnings |

## Understanding

**Status:** ✅ Documented

**Purpose:** Main model class that applies SHiRA adapters to Linear layers by generating sparsity masks and wrapping layers with ShiraLayer implementations.

**Mechanism:** Extends BaseTuner with TRANSFORMERS_MODELS_TO_SHIRA_TARGET_MODULES_MAPPING for defaults. The _create_and_replace method generates mask via config.mask_fn(base_layer, r, **kwargs) before wrapping layers with Linear. The _create_new_module method validates target is torch.nn.Linear (raises ValueError otherwise), generates mask, and creates Linear wrapper. For existing ShiraLayer, calls update_layer with new mask. Passes random_seed when mask_type is "random".

**Significance:** High-level interface for applying SHiRA adapters, orchestrating mask generation and layer replacement. The mask generation happens during model construction, freezing the sparsity pattern for training. Only supports torch.nn.Linear layers, reflecting SHiRA's design for fully-connected adaptations. The architecture enables exploring sparse high-rank alternatives to dense low-rank methods like LoRA.
