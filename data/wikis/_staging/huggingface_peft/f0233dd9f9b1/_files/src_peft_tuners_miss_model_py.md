# File: `src/peft/tuners/miss/model.py`

**Category:** Core Model Implementation

| Property | Value |
|----------|-------|
| Lines | 130 |
| Classes | `MissModel` |
| Imports | layer, peft, torch |

## Understanding

**Status:** Fully explored

**Purpose:** Implements `MissModel` class that creates and manages MiSS (Mixture of Subspaces) Householder reflection adapters for pre-trained models.

**Mechanism:**

### Class Attributes:
- **prefix**: "miss_" (parameter prefix for MiSS adapters)
- **tuner_layer_cls**: `MissLayer` (base layer class)
- **target_module_mapping**: Maps model architectures to default target modules

### Key Method - `_create_and_replace()`:
1. Validates that current_key is not None
2. Extracts bias information from target module
3. Prepares kwargs with MiSS-specific parameters:
   - r (rank along in_features)
   - mini_r (rank along out_features)
   - miss_dropout
   - init_weights (balance/bat/mini mode)
   - bias flag
4. Creates new `MissLinear` module or updates existing `MissLayer`
5. Sets module as non-trainable if adapter not in active_adapters

### Static Method - `_create_new_module()`:
- Currently only supports `torch.nn.Linear` layers
- Creates `MissLinear` instances with adapter configuration
- Raises ValueError for unsupported layer types

**Significance:** MissModel brings Householder reflection-based adaptation to PEFT. Unlike LoRA which factorizes weight updates as low-rank matrices, MiSS uses Householder reflections to create orthogonal transformations. This approach provides:
1. **Matrix-Free Representation**: More memory-efficient than explicit matrices
2. **Orthogonality**: Inherent stability from orthogonal transformations
3. **Flexible Initialization**: Three modes (balance, bat, mini) for different use cases
4. **Competitive Performance**: Achieves good results with fewer parameters

The implementation is based on research from https://huggingface.co/papers/2409.15371 and represents a newer direction in parameter-efficient fine-tuning.

## Key Features

- **Linear Layer Only**: Currently limited to nn.Linear (future support planned)
- **Three Init Modes**: Balance (general), bat (nonlinear), mini (efficient)
- **Householder Reflections**: Orthogonal transformation-based adaptation
- **Bias Support**: Configurable bias training
- **Active Adapter Management**: Controls which adapters are trainable

## Example Usage

```python
from diffusers import StableDiffusionPipeline
from peft import MissModel, MissConfig

config = MissConfig(
    r=64,
    target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
    init_weights=True,  # or "bat" or "mini"
)

model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
model.text_encoder = MissModel(model.text_encoder, config, "default")
```
