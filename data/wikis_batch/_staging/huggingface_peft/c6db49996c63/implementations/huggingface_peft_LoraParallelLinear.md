---
library: huggingface_peft
module: src/peft/tuners/lora/tp_layer.py
classes: ["LoraParallelLinear", "dispatch_megatron"]
type: implementation
tags: ["lora", "megatron", "tensor-parallel", "distributed", "row-parallel", "column-parallel"]
description: Tensor-parallel LoRA implementation for Megatron-LM distributed training
version: c6db49996c63
language: en
---

# LoraParallelLinear: Tensor-Parallel LoRA for Megatron

## Overview

`LoraParallelLinear` implements LoRA for Megatron-LM's tensor-parallel layers, enabling efficient distributed training with LoRA adapters. It supports both `RowParallelLinear` and `ColumnParallelLinear` base layers, intelligently partitioning LoRA matrices to maintain input/output shape consistency across distributed ranks.

Key features:
- **Tensor Parallelism Support**: Works with Megatron's RowParallel and ColumnParallel layers
- **Intelligent Matrix Partitioning**: Automatically splits LoRA_A or LoRA_B based on base layer type
- **FP32 LoRA Precision**: Forces LoRA adapters to FP32 to prevent overflow
- **Standard LoRA Operations**: Supports merge/unmerge, scaling, initialization methods
- **Limitations**: No DoRA, no lora_bias, no Conv1D support

Architecture patterns:
- **RowParallelLinear base**: LoRA_A is row-parallel, LoRA_B is complete linear
- **ColumnParallelLinear base**: LoRA_A is complete linear, LoRA_B is column-parallel

**Source File:** `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/lora/tp_layer.py` (350 lines)

## Code Reference

### LoraParallelLinear Class

Main implementation for tensor-parallel LoRA:

```python
class LoraParallelLinear(nn.Module, LoraLayer):
    """
    Tensor-parallel LoRA layer for Megatron.

    When base layer is RowParallelLinear:
      - lora_A is RowParallelLinear (split by rows)
      - lora_B is standard nn.Linear (complete)

    When base layer is ColumnParallelLinear:
      - lora_A is standard nn.Linear (complete)
      - lora_B is ColumnParallelLinear (split by columns)
    """

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        backend,  # megatron_core.tensor_parallel
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ):
        # Validate unsupported features
        if lora_bias:
            raise ValueError(f"{self.__class__.__name__} does not support lora_bias yet")

        super().__init__()
        LoraLayer.__init__(self, base_layer=base_layer, **kwargs)

        if use_dora:
            raise ValueError(f"{self.__class__.__name__} does not support DoRA yet")

        self.backend = backend
        self.is_parallel_a = isinstance(base_layer, backend.RowParallelLinear)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name

        # Extract Megatron config
        megatron_config = kwargs["megatron_config"]
        parallel_linear_kwargs = {"megatron_config": megatron_config}
        init_method = init.xavier_normal_
        if hasattr(megatron_config, "init_method"):
            init_method = megatron_config.init_method

        # Get parallelism settings from base layer
        input_is_parallel = True
        gather_output = False
        if self.is_parallel_a:
            input_is_parallel = base_layer.input_is_parallel
        else:
            gather_output = base_layer.gather_output

        # Initialize adapter
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            init_method=init_method,
            input_is_parallel=input_is_parallel,
            gather_output=gather_output,
            **parallel_linear_kwargs,
        )

        if is_target_conv_1d_layer:
            raise ValueError(
                f"{self.__class__.__name__} does not support target_conv_1d_layer yet"
            )
        self.is_target_conv_1d_layer = False
```

### Update Layer Method

Creates and configures LoRA adapters with proper parallelization:

```python
def update_layer(
    self,
    adapter_name,
    r,
    lora_alpha,
    lora_dropout,
    init_lora_weights,
    use_rslora,
    use_dora=False,
    init_method=init.xavier_normal_,
    input_is_parallel=True,
    gather_output=False,
    inference_mode: bool = False,
    **parallel_linear_kwargs,
):
    """Update layer with new LoRA adapter"""
    kwargs = locals().copy()
    del kwargs["self"]

    if r <= 0:
        raise ValueError(f"`r` should be positive but got {r}")

    self.r[adapter_name] = r
    self.lora_alpha[adapter_name] = lora_alpha

    # Setup dropout
    if lora_dropout > 0.0:
        lora_dropout_layer = nn.Dropout(p=lora_dropout)
    else:
        lora_dropout_layer = nn.Identity()
    self.lora_dropout[adapter_name] = lora_dropout_layer

    # Force FP32 for LoRA to prevent overflow
    megatron_config = parallel_linear_kwargs["megatron_config"]
    megatron_config.params_dtype = torch.float32

    # Create parallel LoRA layers based on base layer type
    if self.is_parallel_a:
        # RowParallelLinear base: parallelize lora_A
        lora_a = self.backend.RowParallelLinear(
            input_size=self.in_features,
            output_size=r,
            bias=False,
            input_is_parallel=input_is_parallel,
            skip_bias_add=True,
            init_method=init_method,
            config=megatron_config,
        )
        lora_b = nn.Linear(
            in_features=r,
            out_features=self.out_features,
            bias=False,
            dtype=torch.float32
        )
    else:
        # ColumnParallelLinear base: parallelize lora_B
        lora_a = nn.Linear(
            in_features=self.in_features,
            out_features=r,
            bias=False,
            dtype=torch.float32
        )
        lora_b = self.backend.ColumnParallelLinear(
            input_size=r,
            output_size=self.out_features,
            bias=False,
            gather_output=gather_output,
            init_method=init_method,
            config=megatron_config,
        )

    self.lora_A[adapter_name] = lora_a
    self.lora_B[adapter_name] = lora_b

    # Compute scaling
    if use_rslora:
        self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
    else:
        self.scaling[adapter_name] = lora_alpha / r

    self.use_dora[adapter_name] = use_dora

    # Initialize weights
    if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
        with gather_params_ctx(self.get_base_layer().weight):
            self.pissa_init(adapter_name, init_lora_weights)
    elif isinstance(init_lora_weights, str) and init_lora_weights.startswith("corda"):
        with gather_params_ctx(self.get_base_layer().weight):
            self.corda_init(adapter_name, init_lora_weights)
    elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
        with gather_params_ctx(self.get_base_layer().weight):
            self.olora_init(adapter_name)
    elif init_lora_weights == "loftq":
        with gather_params_ctx(self.get_base_layer().weight):
            self.loftq_init(adapter_name)
    elif init_lora_weights:
        self.reset_lora_parameters(adapter_name, init_lora_weights)

    # Move to base layer device
    self._move_adapter_to_device_of_base_layer(adapter_name)

    # Initialize variants (if any)
    if adapter_name in self.lora_variant:
        self.lora_variant[adapter_name].init(self, **kwargs)

    self.set_adapter(self.active_adapters, inference_mode=inference_mode)
```

### Forward Pass

Computes LoRA contribution respecting tensor parallelism:

```python
def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
    """Forward pass combining base layer and LoRA adapters"""
    self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)

    # Call base parallel layer (returns result and bias)
    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result, bias = self.base_layer(x, *args, **kwargs)
    elif adapter_names is not None:
        raise ValueError(f"{self.__class__.__name__} does not support mixed_batch_forward yet")
    elif self.merged:
        result, bias = self.base_layer(x, *args, **kwargs)
    else:
        result, bias = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        # Apply active LoRA adapters
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # Cast input and apply LoRA
            x = self._cast_input_dtype(x, lora_A.weight.dtype)
            result = result + lora_B(lora_A(dropout(x))) * scaling

        result = result.to(torch_result_dtype)

    return result, bias
```

### Merge/Unmerge Operations

Standard LoRA merge with delta weight computation:

```python
def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
    """
    Merge active adapter weights into base weights.

    Args:
        safe_merge: Check for NaNs before merging
        adapter_names: List of adapters to merge (None = all active)
    """
    adapter_names = check_adapters_to_merge(self, adapter_names)
    if not adapter_names:
        return

    for active_adapter in adapter_names:
        if active_adapter in self.lora_A.keys():
            base_layer = self.get_base_layer()

            if safe_merge:
                # Clone and check for NaNs
                orig_weights = base_layer.weight.data.clone()
                delta_weight = self.get_delta_weight(active_adapter)
                orig_weights = orig_weights + delta_weight

                if not torch.isfinite(orig_weights).all():
                    raise ValueError(
                        f"NaNs detected in merged weights for adapter {active_adapter}"
                    )

                base_layer.weight.data = orig_weights
            else:
                delta_weight = self.get_delta_weight(active_adapter)
                base_layer.weight.data = base_layer.weight.data + delta_weight

            self.merged_adapters.append(active_adapter)


def unmerge(self) -> None:
    """Unmerge all merged adapter layers from base weights"""
    if not self.merged:
        warnings.warn("Already unmerged. Nothing to do.")
        return

    while len(self.merged_adapters) > 0:
        active_adapter = self.merged_adapters.pop()
        if active_adapter in self.lora_A.keys():
            weight = self.get_base_layer().weight
            delta_weight = self.get_delta_weight(active_adapter)
            weight.data -= delta_weight


def get_delta_weight(self, adapter) -> torch.Tensor:
    """
    Compute delta weight for given adapter.

    Handles CPU float16/bfloat16 by casting to float32.
    """
    device = self.lora_B[adapter].weight.device
    dtype = self.lora_B[adapter].weight.dtype

    # CPU with fp16/bf16 requires float32 for matmul
    cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

    weight_A = self.lora_A[adapter].weight
    weight_B = self.lora_B[adapter].weight

    if cast_to_fp32:
        weight_A = weight_A.float()
        weight_B = weight_B.float()

    output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

    if cast_to_fp32:
        output_tensor = output_tensor.to(dtype=dtype)
        # Cast back
        self.lora_A[adapter].weight.data = weight_A.to(dtype)
        self.lora_B[adapter].weight.data = weight_B.to(dtype)

    return output_tensor
```

### Dispatcher Function

**dispatch_megatron**: Factory function to create parallel LoRA layers

```python
def dispatch_megatron(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config,
    **kwargs: Any,
) -> Optional[torch.nn.Module]:
    """
    Dispatch function to create LoraParallelLinear for Megatron layers.

    Returns:
        LoraParallelLinear if target is a Megatron parallel layer, None otherwise
    """
    new_module = None

    # Get base layer
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    # Import Megatron if configured
    if lora_config.megatron_config:
        megatron_core = importlib.import_module(lora_config.megatron_core)
    else:
        megatron_core = None

    # Check if target is Megatron parallel layer
    if megatron_core and isinstance(
        target_base_layer,
        (megatron_core.tensor_parallel.ColumnParallelLinear, megatron_core.tensor_parallel.RowParallelLinear),
    ):
        megatron_kwargs = kwargs.copy()
        megatron_config = lora_config.megatron_config

        # Convert dict config to TransformerConfig
        if isinstance(megatron_config, dict):
            transformer_config_class = megatron_core.transformer.transformer_config.TransformerConfig
            megatron_config = transformer_config_class(**lora_config.megatron_config)

        megatron_kwargs["megatron_config"] = megatron_config

        # Disable fan_in_fan_out for parallel layers
        if megatron_kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but target is ColumnParallelLinear or RowParallelLinear. "
                "Setting fan_in_fan_out to False."
            )
            megatron_kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False

        new_module = LoraParallelLinear(
            base_layer=target,
            adapter_name=adapter_name,
            backend=megatron_core.tensor_parallel,
            **megatron_kwargs
        )

    return new_module
```

## I/O Contract

### LoraParallelLinear.__init__()

**Inputs:**
- `base_layer`: Megatron RowParallelLinear or ColumnParallelLinear
- `adapter_name` (str): Name for the adapter
- `backend`: Megatron tensor_parallel module
- `r` (int): LoRA rank
- `lora_alpha` (int): LoRA alpha scaling factor
- `lora_dropout` (float): Dropout probability
- `init_lora_weights` (bool | str): Initialization method
- `use_rslora` (bool): Use rank-stabilized LoRA scaling
- `megatron_config`: Megatron TransformerConfig
- Other standard LoRA parameters

**Outputs:**
- None (creates module in-place)

**Raises:**
- `ValueError`: If lora_bias=True (unsupported)
- `ValueError`: If use_dora=True (unsupported)
- `ValueError`: If is_target_conv_1d_layer=True (unsupported)

**Side Effects:**
- Creates lora_A and lora_B with appropriate parallelization
- Forces params_dtype to float32 in megatron_config
- Registers adapter in internal dictionaries

### forward()

**Inputs:**
- `x` (torch.Tensor): Input tensor, shape `(batch, seq_len, in_features)`
- `*args`, `**kwargs`: Additional arguments passed to base layer

**Outputs:**
- `result` (torch.Tensor): Output tensor
- `bias` (torch.Tensor | None): Bias term (if base layer has bias)

**Behavior:**
- Calls base parallel layer first
- Adds LoRA contribution for each active adapter
- Returns tuple `(result, bias)` following Megatron convention

**Constraints:**
- Does not support `adapter_names` kwarg (mixed batch forward)
- Automatically handles tensor parallel communication

### get_delta_weight()

**Inputs:**
- `adapter` (str): Adapter name to compute delta for

**Outputs:**
- `torch.Tensor`: Delta weight, shape `(out_features, in_features)`

**Computational Notes:**
- On CPU with fp16/bf16, casts to fp32 for matmul
- Applies fan_in_fan_out transpose if needed
- Includes scaling factor in output

### dispatch_megatron()

**Inputs:**
- `target` (torch.nn.Module): Target module to wrap
- `adapter_name` (str): Adapter name
- `lora_config`: LoraConfig with megatron_config set
- `**kwargs`: Additional LoRA parameters

**Outputs:**
- `LoraParallelLinear | None`: New module if target is Megatron parallel layer, None otherwise

**Side Effects:**
- May modify `lora_config.fan_in_fan_out` to False
- Emits warning if fan_in_fan_out was True

## Usage Examples

### Basic Megatron LoRA Setup

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
import megatron.core as mcore

# Initialize Megatron
megatron_config = mcore.transformer.transformer_config.TransformerConfig(
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    tensor_model_parallel_size=4,  # 4-way tensor parallelism
)

# Load model (should have Megatron parallel layers)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure LoRA with Megatron support
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    megatron_config=megatron_config,
    megatron_core="megatron.core",  # Import path
)

# Create PEFT model
peft_model = get_peft_model(model, config)

# LoRA adapters automatically created as parallel layers
```

### Checking Parallel Layer Types

```python
# Inspect which layers are parallelized
for name, module in peft_model.named_modules():
    if hasattr(module, 'lora_A'):
        for adapter_name, lora_a in module.lora_A.items():
            lora_b = module.lora_B[adapter_name]
            print(f"{name}:")
            print(f"  LoRA A type: {type(lora_a).__name__}")
            print(f"  LoRA B type: {type(lora_b).__name__}")
            print(f"  Base layer type: {type(module.get_base_layer()).__name__}")
```

### Multi-GPU Training with Tensor Parallelism

```python
import torch.distributed as dist
from megatron.core import parallel_state

# Initialize Megatron distributed
parallel_state.initialize_model_parallel(tensor_model_parallel_size=4)

# Setup model with LoRA
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    megatron_config=megatron_config,
    megatron_core="megatron.core",
)

model = get_peft_model(base_model, config)

# Training loop
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    # Megatron handles gradient synchronization across tensor-parallel ranks
```

### Using Different Initialization Methods

```python
# PiSSA initialization with tensor parallelism
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    init_lora_weights="pissa",  # Supported!
    megatron_config=megatron_config,
    megatron_core="megatron.core",
)

peft_model = get_peft_model(model, config)

# Other supported initializations:
# - "pissa", "pissa_niter_[n]"
# - "corda", "corda_ipm", "corda_kpm"
# - "olora"
# - "loftq"
# - True (default Kaiming)
```

### Merging Adapters (Single GPU)

```python
# Note: Merging should be done before distributing across GPUs
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj"],
    megatron_config=megatron_config,
    megatron_core="megatron.core",
)

peft_model = get_peft_model(model, config)

# Train...

# Merge adapters (do this on single GPU before saving)
peft_model.merge_adapter()

# Now save merged model
peft_model.save_pretrained("./merged_model")
```

### Handling Multiple Adapters

```python
# Load first adapter
config1 = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    megatron_config=megatron_config,
    megatron_core="megatron.core",
)
peft_model = get_peft_model(model, config1, adapter_name="adapter1")

# Add second adapter
config2 = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    megatron_config=megatron_config,
    megatron_core="megatron.core",
)
peft_model.add_adapter("adapter2", config2)

# Switch between adapters
peft_model.set_adapter("adapter1")
outputs1 = peft_model(**inputs)

peft_model.set_adapter("adapter2")
outputs2 = peft_model(**inputs)
```

### Verifying FP32 Precision

```python
# LoRA weights are forced to FP32 even if base model is FP16
for name, module in peft_model.named_modules():
    if hasattr(module, 'lora_A'):
        for adapter_name in module.lora_A.keys():
            lora_a_dtype = module.lora_A[adapter_name].weight.dtype
            lora_b_dtype = module.lora_B[adapter_name].weight.dtype
            print(f"{name}.{adapter_name}:")
            print(f"  LoRA A dtype: {lora_a_dtype}")  # torch.float32
            print(f"  LoRA B dtype: {lora_b_dtype}")  # torch.float32
            print(f"  Base dtype: {module.get_base_layer().weight.dtype}")
```

### Custom Megatron Config

```python
# Define custom Megatron configuration
custom_config = mcore.transformer.transformer_config.TransformerConfig(
    num_layers=24,
    hidden_size=2048,
    num_attention_heads=16,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=2,
    params_dtype=torch.bfloat16,  # Will be overridden to FP32 for LoRA
    init_method=torch.nn.init.xavier_uniform_,
)

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    megatron_config=custom_config,
    megatron_core="megatron.core",
)

peft_model = get_peft_model(model, config)
```

### Error Handling

```python
try:
    # This will raise an error - DoRA not supported
    config = LoraConfig(
        r=16,
        use_dora=True,
        megatron_config=megatron_config,
        megatron_core="megatron.core",
    )
    peft_model = get_peft_model(model, config)
except ValueError as e:
    print(f"Error: {e}")  # "LoraParallelLinear does not support DoRA yet"

try:
    # This will raise an error - lora_bias not supported
    config = LoraConfig(
        r=16,
        lora_bias=True,
        megatron_config=megatron_config,
        megatron_core="megatron.core",
    )
    peft_model = get_peft_model(model, config)
except ValueError as e:
    print(f"Error: {e}")  # "LoraParallelLinear does not support lora_bias yet"
```

## Related Pages

### Core LoRA Components
- `huggingface_peft_LoraLayer.md` - Base LoRA layer implementation
- `huggingface_peft_Linear.md` - Standard LoRA linear layer
- `huggingface_peft_LoraConfig.md` - Configuration including megatron_config

### Megatron Integration
- Megatron-LM documentation: Tensor and pipeline parallelism
- `torch.distributed` - PyTorch distributed training

### Initialization Methods
- `huggingface_peft_pissa_init.md` - PiSSA initialization
- `huggingface_peft_corda_init.md` - CorDA initialization
- `huggingface_peft_olora_init.md` - OLoRA initialization
- `huggingface_peft_loftq_init.md` - LoftQ initialization

### Utilities
- `huggingface_peft_gather_params_ctx.md` - Context manager for gathering distributed parameters
- `huggingface_peft_transpose.md` - Weight transposition utilities
- `huggingface_peft_check_adapters_to_merge.md` - Merge validation

### Concepts
- Tensor parallelism in large language models
- RowParallel vs ColumnParallel layer partitioning
- Distributed LoRA training strategies
- Precision considerations in distributed training
