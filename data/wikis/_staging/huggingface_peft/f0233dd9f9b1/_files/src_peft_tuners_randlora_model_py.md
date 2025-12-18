# File: `src/peft/tuners/randlora/model.py`

**Category:** model

| Property | Value |
|----------|-------|
| Lines | 357 |
| Classes | `RandLoraModel` |
| Functions | `_kaiming_init` |
| Imports | __future__, _buffer_dict, accelerate, config, layer, math, peft, torch, transformers, tuners_utils, typing, warnings |

## Understanding

**Status:** ✅ Fully explored

**Purpose:** Main model class managing RandLora adapters with shared frozen random projections and per-layer trainable scaling.

**Mechanism:**
- **_kaiming_init() function**: Custom Kaiming uniform initialization accepting torch.Generator for deterministic PRNG
  - Supports tensor_or_shape input (tensor to init or shape to create)
  - Uses bf16 if available, else fp16
  - Applies standard Kaiming uniform with a=√5

- **RandLoraModel class** extends `BaseTuner`:
  - `prefix = "randlora_"`, `tuner_layer_cls = RandLoraLayer`
  - Manages shared random bases `randlora_A` and `randlora_B` using BufferDict

- **Key Methods:**
  - `_find_dim()`: Finds largest input/output dimensions across all target linear layers to size shared bases
  - `_init_randlora_A_randlora_B()`: Initializes dense random bases:
    - `randlora_A`: (r, 1, min_dim) - applied to smallest dimension
    - `randlora_B`: (max_dim, num_bases, r) - ensures full rank
    - num_bases = ceil(min_dim / r) to guarantee full rank
    - Uses Kaiming init then std normalization
  - `_init_randlora_A_randlora_B_sparse()`: Initializes sparse ternary bases {-1, 0, 1}
    - Standard sparse: 1/(2*sparsity) probability for ±1, rest is 0
    - Very sparse: 1/√D probability for ±1, rest is 0
    - Std normalization applied after sparsification
  - `_pre_injection_hook()`: Called before adapter injection to initialize shared bases
  - `_check_new_adapter_config()`: Validates new adapters have same projection_prng_key and save_projection settings
  - `_create_and_replace()`: Creates RandLora layers or updates existing ones
  - `_create_new_module()`: Factory method supporting Linear, Conv1D, Linear8bitLt, and Linear4bit

- **Shared Basis Architecture:**
  - All RandLora layers share the same randlora_A and randlora_B bases
  - Bases are frozen (non-trainable)
  - Each layer has trainable lambda and gamma diagonal matrices for scaling
  - Persistence controlled by save_projection flag

**Significance:** Core orchestrator implementing RandLora's key innovation: using shared frozen random projections across all layers while maintaining per-layer trainable scaling. This dramatically reduces parameters compared to LoRA:
- LoRA: O(layers × rank × dims) trainable parameters
- RandLora: O(rank × max_dims) frozen + O(layers × bases × min_dim) trainable
The shared random bases are initialized once using deterministic PRNG, ensuring reproducibility. The sparse variants enable future matmul-free optimizations while potentially reducing overfitting.
