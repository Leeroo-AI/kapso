# File: `src/peft/tuners/gralora/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 182 |
| Classes | `GraloraConfig` |
| Imports | dataclasses, peft, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines GraloraConfig, the configuration dataclass for GraLoRA (Granular Low-Rank Adaptation), which specifies hyperparameters for block-wise low-rank weight adaptation with optional hybrid LoRA components.

**Mechanism:** GraloraConfig stores key parameters: (1) r - the total rank for GraLoRA (default 32), which is divided across gralora_k subblocks; (2) gralora_k - number of subblocks (default 2 for r≤32, recommended 4 for r≥64), where r must be divisible by gralora_k; (3) hybrid_r - rank for optional vanilla LoRA component (default 0), enabling "Hybrid GraLoRA" when >0; (4) alpha - scaling factor where final scale = alpha/(r+hybrid_r); (5) gralora_dropout - dropout probability; (6) Standard PEFT parameters like target_modules, fan_in_fan_out, bias, init_weights, layers_to_transform, and layers_pattern. The __post_init__ validates r divisibility by gralora_k and converts target_modules to set format.

**Significance:** This configuration enables GraLoRA, an advanced LoRA variant that achieves higher expressivity by dividing matrices into gralora_k subblocks and enabling information exchange between them. With the same parameter count as LoRA rank r, GraLoRA's expressivity is effectively multiplied by gralora_k. Hybrid GraLoRA combines GraLoRA's block-wise updates with vanilla LoRA's global updates for even better performance at the cost of r+hybrid_r parameters.
