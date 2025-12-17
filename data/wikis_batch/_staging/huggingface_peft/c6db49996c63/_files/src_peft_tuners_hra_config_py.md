# File: `src/peft/tuners/hra/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 133 |
| Classes | `HRAConfig` |
| Imports | __future__, dataclasses, peft, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines configuration parameters for HRA (Householder Reflection Adaptation) orthogonal transformation method.

**Mechanism:** HRAConfig extends PeftConfig with HRA parameters: r (rank/number of reflections, best as even number for symmetric initialization), apply_GS (enable Gram-Schmidt orthogonalization), target_modules, exclude_modules, init_weights (controls initialization strategy), layer selection options, and bias settings. Validates layer pattern constraints in __post_init__.

**Significance:** Core configuration for Householder reflection-based adaptation. Paper reference: https://huggingface.co/papers/2405.17484. The r parameter controls adaptation capacity via number of sequential reflections. Gram-Schmidt orthogonalization provides alternative to iterative reflection composition. Supports both transformer (Linear) and vision (Conv2d) models.
