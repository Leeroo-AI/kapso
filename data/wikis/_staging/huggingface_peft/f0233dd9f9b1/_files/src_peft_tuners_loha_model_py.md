# File: `src/peft/tuners/loha/model.py`

**Category:** Core Model Implementation

| Property | Value |
|----------|-------|
| Lines | 116 |
| Classes | `LoHaModel` |
| Imports | layer, peft, torch, typing |

## Understanding

**Status:** Fully explored

**Purpose:** Implements the `LoHaModel` class, which creates and manages Low-Rank Hadamard Product adapters for pre-trained neural network models.

**Mechanism:**

The `LoHaModel` class extends `LycorisTuner` and provides the core functionality for applying LoHa adapters:

### Class Attributes:
- **prefix**: "hada_" (Hadamard product parameter prefix)
- **tuner_layer_cls**: `LoHaLayer` (base layer class for adapters)
- **target_module_mapping**: Maps model architectures to default target modules
- **layers_mapping**: Maps PyTorch layer types to LoHa implementations:
  - `torch.nn.Linear` → `Linear`
  - `torch.nn.Conv2d` → `Conv2d`
  - `torch.nn.Conv1d` → `Conv1d`

### Key Method - `_create_and_replace()`:
1. Retrieves rank and alpha values using pattern matching (`get_pattern_key`)
2. Applies custom rank/alpha if patterns match the current layer key
3. Either updates existing `LoHaLayer` or creates new adapter module
4. Replaces the target module in the parent container

### Initialization:
Creates LoHa adapters from pre-trained models with support for:
- Multiple adapter types (Linear, Conv1d, Conv2d)
- Pattern-based rank/alpha customization per layer
- Low CPU memory usage mode (meta device initialization)
- Diffusion model integration (e.g., Stable Diffusion)

**Significance:** This is the main entry point for applying LoHa adapters to models. LoHa uses the Hadamard product (element-wise multiplication) of two low-rank decompositions, providing a more expressive alternative to LoRA while maintaining parameter efficiency. The model supports both transformer-based LLMs and diffusion models, making it versatile for various deep learning tasks. The implementation is based on research from https://huggingface.co/papers/2108.06098 and borrows from the LyCORIS library.

## Key Features

- **Multi-Layer Support**: Linear, Conv1d, and Conv2d layers
- **Pattern-Based Customization**: Per-layer rank and alpha configuration
- **Diffusion Model Ready**: Examples show integration with Stable Diffusion
- **LyCORIS Integration**: Part of the broader LyCORIS adapter family
- **Dynamic Module Creation**: Adapts based on layer type and configuration

## Example Usage

```python
from diffusers import StableDiffusionPipeline
from peft import LoHaModel, LoHaConfig

config_te = LoHaConfig(
    r=8,
    lora_alpha=32,
    target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
    rank_dropout=0.0,
    module_dropout=0.0,
)

model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
model.text_encoder = LoHaModel(model.text_encoder, config_te, "default")
```
