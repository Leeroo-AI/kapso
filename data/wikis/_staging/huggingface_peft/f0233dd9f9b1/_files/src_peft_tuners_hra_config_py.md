# File: `src/peft/tuners/hra/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 133 |
| Classes | `HRAConfig` |
| Imports | __future__, dataclasses, peft, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines HRAConfig, the configuration dataclass for HRA (Householder Reflection Adaptation), which specifies hyperparameters for applying orthogonal transformations to pretrained weights using Householder reflections.

**Mechanism:** HRAConfig stores key parameters: (1) r - the rank/number of Householder reflections to apply (default 8, recommended to be even for symmetric initialization); (2) apply_GS - whether to apply Gram-Schmidt orthogonalization to ensure orthogonality (default False); (3) target_modules - which layers to adapt (Linear/Conv2d), supporting regex patterns or 'all-linear' wildcard; (4) exclude_modules - layers to exclude from adaptation; (5) init_weights - whether to use default initialization (True) or random initialization (False); (6) Standard PEFT parameters like layers_to_transform, layers_pattern, bias ('none', 'all', 'hra_only'), and modules_to_save. The __post_init__ validates that layers_to_transform and layers_pattern are only used with list-based target_modules, not regex patterns.

**Significance:** This configuration enables HRA (https://huggingface.co/papers/2405.17484), a method that uses r Householder reflections to create orthogonal weight transformations. Each reflection is parameterized by a single vector u, making HRA very parameter-efficient (r vectors of size d). Unlike low-rank methods that compress information, HRA preserves the full rank while rotating the weight space. The apply_GS option ensures mathematical orthogonality but is typically unnecessary with proper initialization.
