# RandLoraModel (RandLoRA Model Manager)

## Implementation Overview

**File:** `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/randlora/model.py`
**Lines of Code:** 356
**Language:** Python

RandLoraModel orchestrates the creation and management of RandLoRA adapters, handling the initialization of shared random projection bases and their distribution across target layers.

## Core Implementation

### Model Class

```python
class RandLoraModel(BaseTuner):
    """Creates RandLoRA model from pretrained transformers model

    Args:
        model: Model to adapt
        config: RandLoraConfig
        adapter_name: Adapter name (default: "default")
        low_cpu_mem_usage: Use meta device for initialization

    Key Feature: Shared random bases (randlora_A, randlora_B) across all layers
    """

    prefix: str = "randlora_"
    tuner_layer_cls = RandLoraLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_RANDLORA_TARGET_MODULES_MAPPING
```

## Key Methods

### Dimension Finding

**Method:** `_find_dim()`

Determines the maximum dimensions needed for shared bases:

```python
def _find_dim(self, config) -> tuple[int, int]:
    """Find largest input/output dimensions across all target layers"""
    model_config = self.get_model_config(self.model)
    peft_config = self._prepare_adapter_config(config, model_config)
    peft_config = _maybe_include_all_linear_layers(peft_config, self.model)

    largest_shape = None
    for key, module in self.model.named_modules():
        if not self._check_target_module_exists(peft_config, key):
            continue

        if isinstance(module, nn.Linear):
            module_shape = module.out_features, module.in_features
        elif isinstance(module, Conv1D):
            module_shape = module.weight.ds_shape if hasattr(module.weight, "ds_shape") else module.weight.shape
            module_shape = module_shape[::-1]
        else:
            continue

        if largest_shape is None:
            largest_shape = module_shape
            continue

        if module_shape != largest_shape:
            largest_shape = tuple(max(a, b) for a, b in zip(largest_shape, module_shape))

    if largest_shape is None:
        raise ValueError("No RandLoRA-compatible layers found")

    return largest_shape
```

**Purpose:** Ensures shared bases are large enough for all layers

### Base Initialization (Dense)

**Method:** `_init_randlora_A_randlora_B()`

Initializes dense random projection bases:

```python
def _init_randlora_A_randlora_B(self, config: RandLoraConfig, adapter_name: str) -> None:
    linear_out_dim, linear_in_dim = self._find_dim(config)
    max_dim, min_dim = max(linear_out_dim, linear_in_dim), min(linear_out_dim, linear_in_dim)

    self.randlora_A = BufferDict({}, persistent=config.save_projection)
    self.randlora_B = BufferDict({}, persistent=config.save_projection)

    # Deterministic initialization
    generator = torch.Generator(device="cpu").manual_seed(config.projection_prng_key)

    # A is smallest matrix (min_dim) to reduce trainable params
    randlora_A = _kaiming_init((config.r, 1, min_dim), generator=generator)

    # B is largest matrix, ensure full rank
    num_bases = min(linear_out_dim, linear_in_dim) / config.r
    num_bases = int(num_bases) if num_bases.is_integer() else int(num_bases) + 1
    randlora_B = torch.cat(
        [_kaiming_init((max_dim, 1, config.r), generator=generator) for _ in range(num_bases)], dim=1
    )

    # Std normalization (empirically best)
    randlora_A, randlora_B = randlora_A / randlora_A.std(), randlora_B / randlora_B.std()
    self.randlora_A[adapter_name] = randlora_A
    self.randlora_B[adapter_name] = randlora_B
```

**Key Points:**
- Uses Kaiming initialization
- Normalizes by standard deviation
- Deterministic via PRNG seed
- Stored as buffers (non-trainable)

### Base Initialization (Sparse)

**Method:** `_init_randlora_A_randlora_B_sparse()`

Initializes sparse ternary bases:

```python
def _init_randlora_A_randlora_B_sparse(self, config: RandLoraConfig, adapter_name: str, sparsity: int = 3) -> None:
    linear_out_dim, linear_in_dim = self._find_dim(config)
    max_dim, min_dim = max(linear_out_dim, linear_in_dim), min(linear_out_dim, linear_in_dim)

    self.randlora_A = BufferDict({}, persistent=config.save_projection)
    self.randlora_B = BufferDict({}, persistent=config.save_projection)

    generator = torch.Generator(device="cpu").manual_seed(config.projection_prng_key)

    randlora_A = torch.rand((config.r, 1, min_dim), generator=generator)
    num_bases = min_dim / config.r
    num_bases = int(num_bases) if num_bases.is_integer() else int(num_bases) + 1
    randlora_B = torch.rand((max_dim, num_bases, config.r), generator=generator)

    # Create ternary sparse matrices
    randlora_B_sparse = torch.zeros(randlora_B.shape)
    randlora_A_sparse = torch.zeros(randlora_A.shape)
    randlora_B_sparse[randlora_B < 1 / (2 * sparsity)] = -1
    randlora_B_sparse[randlora_B > 1 - 1 / (2 * sparsity)] = 1
    randlora_A_sparse[randlora_A < 1 / (2 * sparsity)] = -1
    randlora_A_sparse[randlora_A > 1 - 1 / (2 * sparsity)] = 1

    # Std normalization
    randlora_A, randlora_B = (
        randlora_A_sparse / randlora_A_sparse.std(),
        randlora_B_sparse / randlora_B_sparse.std(),
    )
    self.randlora_A[adapter_name] = randlora_A
    self.randlora_B[adapter_name] = randlora_B
```

**Sparsity Levels:**
- `sparsity=3`: P(-1)=1/6, P(0)=2/3, P(1)=1/6
- `sparsity=√D`: Very sparse variant

### Pre-Injection Hook

**Method:** `_pre_injection_hook()`

Called before adapter injection to initialize bases:

```python
def _pre_injection_hook(self, model: nn.Module, config: RandLoraConfig, adapter_name: str) -> None:
    if config.very_sparse:
        linear_out_dim, linear_in_dim = self._find_dim(config)
        self._init_randlora_A_randlora_B_sparse(
            config, adapter_name, sparsity=math.sqrt(min(linear_out_dim, linear_in_dim))
        )
    elif config.sparse:
        self._init_randlora_A_randlora_B_sparse(config, adapter_name, sparsity=3)
    else:
        self._init_randlora_A_randlora_B(config, adapter_name)
```

**Initialization Strategy:**
- very_sparse → sparsity = √D
- sparse → sparsity = 3
- default → dense Kaiming

### Layer Creation

**Method:** `_create_and_replace()`

Creates RandLoRA layers and replaces target modules:

```python
def _create_and_replace(
    self,
    randlora_config,
    adapter_name,
    target,
    target_name,
    parent,
    current_key,
    **optional_kwargs,
):
    if current_key is None:
        raise ValueError("Current Key shouldn't be `None`")

    r = randlora_config.r
    bias = hasattr(target, "bias") and target.bias is not None
    kwargs = {
        "r": r,
        "randlora_alpha": randlora_config.randlora_alpha,
        "randlora_dropout": randlora_config.randlora_dropout,
        "fan_in_fan_out": randlora_config.fan_in_fan_out,
        "init_weights": randlora_config.init_weights,
        "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
        "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
    }
    kwargs["bias"] = bias

    if isinstance(target, Linear):
        # Update existing RandLoRA layer
        target.update_layer(
            adapter_name,
            self.randlora_A,
            self.randlora_B,
            r,
            randlora_config.randlora_alpha,
            randlora_config.randlora_dropout,
            randlora_config.init_weights,
        )
    else:
        # Create new RandLoRA layer
        new_module = self._create_new_module(
            randlora_config, self.randlora_A, self.randlora_B, adapter_name, target, **kwargs
        )
        if adapter_name not in self.active_adapter:
            new_module.requires_grad_(False)
        self._replace_module(parent, target_name, new_module, target)
```

### Module Factory

**Method:** `_create_new_module()`

Creates appropriate RandLoRA module based on layer type:

```python
@staticmethod
def _create_new_module(randlora_config, randlora_A, randlora_B, adapter_name, target, **kwargs):
    # Handle quantized layers
    if is_bnb_available():
        import bitsandbytes as bnb
        from .bnb import Linear8bitLt

    if is_bnb_4bit_available():
        from .bnb import Linear4bit

    bias = kwargs.pop("bias", False)
    loaded_in_8bit = kwargs.get("loaded_in_8bit", False)
    loaded_in_4bit = kwargs.get("loaded_in_4bit", False)

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    # Create appropriate module type
    if loaded_in_8bit and isinstance(target_base_layer, bnb.nn.Linear8bitLt):
        eightbit_kwargs = kwargs.copy()
        eightbit_kwargs.update({
            "has_fp16_weights": target_base_layer.state.has_fp16_weights,
            "threshold": target_base_layer.state.threshold,
            "index": target_base_layer.index,
        })
        return Linear8bitLt(target, adapter_name, randlora_A, randlora_B, **eightbit_kwargs)
    elif loaded_in_4bit and isinstance(target_base_layer, bnb.nn.Linear4bit):
        fourbit_kwargs = kwargs.copy()
        fourbit_kwargs.update({
            "compute_dtype": target_base_layer.compute_dtype,
            "compress_statistics": target_base_layer.weight.compress_statistics,
            "quant_type": target_base_layer.weight.quant_type,
        })
        return Linear4bit(target, adapter_name, randlora_A, randlora_B, **fourbit_kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn("fan_in_fan_out set to True but target is torch.nn.Linear. Setting to False.")
            kwargs["fan_in_fan_out"] = randlora_config.fan_in_fan_out = False
    elif isinstance(target_base_layer, Conv1D):
        kwargs["is_target_conv_1d_layer"] = True
        if not kwargs["fan_in_fan_out"]:
            warnings.warn("fan_in_fan_out set to False but target is Conv1D. Setting to True.")
            kwargs["fan_in_fan_out"] = randlora_config.fan_in_fan_out = True
    else:
        raise ValueError(f"Target module {target} not supported. Only Linear and Conv1D supported.")

    new_module = Linear(
        target,
        randlora_A,
        randlora_B,
        adapter_name,
        bias=bias,
        **kwargs,
    )

    return new_module
```

### Configuration Validation

**Method:** `_check_new_adapter_config()`

Validates new adapter configuration:

```python
def _check_new_adapter_config(self, config: RandLoraConfig) -> None:
    super()._check_new_adapter_config(config)

    # Check PRNG key consistency
    for existing_config in self.peft_config.values():
        if existing_config is config:
            continue

        if existing_config.projection_prng_key != config.projection_prng_key:
            raise ValueError(
                f"PRNG key must be same for all adapters. Got {config.projection_prng_key} "
                f"but previous config had {existing_config.projection_prng_key}."
            )

    # Check save_projection consistency
    save_project_unique_values = sorted({config.save_projection for config in self.peft_config.values()})
    if len(save_project_unique_values) > 1:
        raise ValueError(
            "save_projection must be same for all adapters. Got multiple values: "
            f"{save_project_unique_values}"
        )
```

## Helper Function

### Kaiming Initialization

```python
def _kaiming_init(
    tensor_or_shape: Union[torch.Tensor, tuple[int, ...]],
    generator: torch.Generator,
) -> torch.Tensor:
    """Kaiming Uniform Init with Generator for PRNG"""
    if isinstance(tensor_or_shape, tuple):
        tensor = torch.empty(
            tensor_or_shape,
            dtype=torch.bfloat16 if is_bf16_available() else torch.float16,
        )
    else:
        tensor = tensor_or_shape

    with torch.no_grad():
        basis = torch.nn.init.kaiming_uniform_(tensor, a=math.sqrt(5), generator=generator)
        return basis
```

## Usage Example

```python
from transformers import AutoModelForCausalLM
from peft import RandLoraConfig, get_peft_model

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# Create RandLoRA configuration
config = RandLoraConfig(r=32)

# Apply RandLoRA (automatically initializes shared bases)
model = get_peft_model(base_model, config)

# Train
model.train()
for batch in dataloader:
    output = model(batch)
    loss.backward()
    optimizer.step()
```

## Design Patterns

### Singleton Pattern (Shared Bases)

```python
# Single shared bases for entire model
self.randlora_A = BufferDict({}, persistent=config.save_projection)
self.randlora_B = BufferDict({}, persistent=config.save_projection)
```

### Factory Pattern

```python
@staticmethod
def _create_new_module(randlora_config, randlora_A, randlora_B, ...):
    # Creates appropriate module based on target type
```

### Hook Pattern

```python
def _pre_injection_hook(self, model, config, adapter_name):
    # Called before injection to initialize shared bases
```

## References

- **Paper**: https://huggingface.co/papers/2502.00987
- **Type**: `PeftType.RANDLORA`
- **Key Feature**: Shared random projection bases
- **Efficiency**: 2-3x parameter reduction vs LoRA
