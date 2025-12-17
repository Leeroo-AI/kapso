---
library: huggingface_peft
module: src/peft/tuners/lora/variants.py
classes: ["DoraLinearVariant", "DoraEmbeddingVariant", "DoraConv1dVariant", "DoraConv2dVariant", "DoraConv3dVariant", "QALoraLinearVariant", "ALoraLinearVariant", "BdLoraLinearVariant", "BlockDiagonalLinear"]
type: implementation
tags: ["lora", "variants", "dora", "qalora", "alora", "bdlora", "block-diagonal", "adapter"]
description: Complete collection of LoRA variant adapters including DoRA, QALoRA, aLoRA, and Block-Diagonal LoRA
version: c6db49996c63
language: en
---

# LoRA Variants Collection

## Overview

The `variants.py` module contains the complete collection of LoRA variant implementations that extend the base LoRA functionality with specialized techniques. Each variant implements the `LoraVariant` interface, providing custom initialization, forward pass, and merge/unmerge operations.

**Variants included:**
1. **DoRA (Weight-Decomposed LoRA)**: Magnitude-direction decomposition for better expressiveness
2. **QALoRA (Quantization-Aware LoRA)**: Pooling-based LoRA for quantized models
3. **aLoRA (Activated LoRA)**: Token-selective activation based on invocation sequences
4. **BdLoRA (Block-Diagonal LoRA)**: Structured sparsity with block-diagonal matrices
5. **Arrow**: MoE adaptive routing (covered in separate doc)

All variants follow the LoraVariant protocol with `init()`, `forward()`, `merge_safe()`, `merge_unsafe()`, and `unmerge()` static methods.

**Source File:** `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/lora/variants.py` (926 lines)

## Code Reference

### DoRA Variants

DoRA variants decompose weight updates into magnitude and direction:

#### DoraLinearVariant

```python
class DoraLinearVariant(LoraVariant):
    @staticmethod
    def init(module: Linear, adapter_name: str, **kwargs: Any) -> None:
        """Initialize DoRA magnitude vector"""
        if not module.lora_magnitude_vector:
            # First DoRA layer: add to learnable parameters
            module.adapter_layer_names = module.adapter_layer_names[:] + ("lora_magnitude_vector",)

        # Create DoRA layer
        dora_layer = DoraLinearLayer(fan_in_fan_out=getattr(module, "fan_in_fan_out", False))

        # Get LoRA weights
        lora_A = module.lora_A[adapter_name].weight
        lora_B = module.lora_B[adapter_name].weight

        # Handle ephemeral GPU offload
        place_on_cpu = module.ephemeral_gpu_offload and (lora_A.device.type == "cpu" or lora_B.device.type == "cpu")
        if module.ephemeral_gpu_offload:
            # Move to appropriate device
            if lora_A.device.type in ["cuda", "xpu"]:
                lora_B = lora_B.to(lora_A.device)
            else:
                if lora_B.device.type not in ["cuda", "xpu"]:
                    lora_B = lora_B.to("xpu" if is_xpu_available() else "cuda")
                lora_A = lora_A.to(lora_B.device)

        scaling = module.scaling[adapter_name]

        # Initialize DoRA layer
        dora_layer.update_layer(
            base_layer=module.get_base_layer(),
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=scaling,
            place_on_cpu=place_on_cpu,
        )

        module.lora_magnitude_vector[adapter_name] = dora_layer

    @staticmethod
    def forward(
        module: Linear,
        active_adapter: str,
        x: torch.Tensor,
        result: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """DoRA forward with magnitude-direction decomposition"""
        lora_A = module.lora_A[active_adapter]
        lora_B = module.lora_B[active_adapter]
        dropout = module.lora_dropout[active_adapter]
        scaling = module.scaling[active_adapter]

        # Apply dropout if training
        if isinstance(dropout, nn.Identity) or not module.training:
            base_result = result
        else:
            x = dropout(x)
            base_result = None

        # Apply DoRA
        result = result + module.lora_magnitude_vector[active_adapter](
            x,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=scaling,
            base_layer=module.get_base_layer(),
            base_result=base_result,
        )
        return result

    @staticmethod
    def merge_safe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        """Safe merge with DoRA magnitude scaling"""
        orig_dtype = orig_weight.dtype
        delta_weight = module.get_delta_weight(active_adapter)

        # Compute weight norm (scaling already in delta_weight, so set to 1)
        weight_norm = (
            module.lora_magnitude_vector[active_adapter]
            .get_weight_norm(orig_weight, transpose(delta_weight, module.fan_in_fan_out), scaling=1)
            .detach()
        )

        # Cache weight_norm for unmerge
        module._cache_store(f"{active_adapter}-weight_norm", weight_norm)

        # Apply DoRA magnitude scaling
        dora_factor = module.lora_magnitude_vector[active_adapter].weight / weight_norm
        dora_factor = transpose(dora_factor.view(-1, 1), module.fan_in_fan_out)
        new_weight = dora_factor * (orig_weight + delta_weight)
        new_weight = new_weight.to(orig_dtype)
        return new_weight

    @staticmethod
    def merge_unsafe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> None:
        """Unsafe in-place merge"""
        orig_dtype = orig_weight.dtype
        delta_weight = module.get_delta_weight(active_adapter)

        weight_norm = (
            module.lora_magnitude_vector[active_adapter]
            .get_weight_norm(orig_weight, transpose(delta_weight, module.fan_in_fan_out), scaling=1)
            .detach()
        )
        module._cache_store(f"{active_adapter}-weight_norm", weight_norm)

        dora_factor = module.lora_magnitude_vector[active_adapter].weight / weight_norm
        dora_factor = transpose(dora_factor.view(-1, 1), module.fan_in_fan_out)
        new_weight = dora_factor * (orig_weight.data + delta_weight)
        orig_weight.data = new_weight.to(orig_dtype)

    @staticmethod
    def unmerge(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        """Unmerge DoRA adapter"""
        orig_dtype = orig_weight.dtype
        delta_weight = module.get_delta_weight(active_adapter)

        # Retrieve cached weight_norm
        weight_norm = module._cache_pop(f"{active_adapter}-weight_norm")
        dora_factor = module.lora_magnitude_vector[active_adapter].weight / weight_norm

        # Reverse DoRA transformation
        new_weight = orig_weight.data / dora_factor.view(-1, 1) - delta_weight
        new_weight = new_weight.to(orig_dtype)
        return new_weight
```

#### Other DoRA Variants

```python
class DoraEmbeddingVariant(DoraLinearVariant):
    """DoRA for embedding layers - similar structure with embedding-specific logic"""

class _DoraConvNdVariant(LoraVariant):
    """Base class for convolutional DoRA variants"""

class DoraConv1dVariant(_DoraConvNdVariant):
    """DoRA for Conv1d layers"""

class DoraConv2dVariant(_DoraConvNdVariant):
    """DoRA for Conv2d layers"""

class DoraConv3dVariant(_DoraConvNdVariant):
    """DoRA for Conv3d layers"""
```

### QALoRA (Quantization-Aware LoRA)

QALoRA uses input pooling to reduce computational cost with quantized models:

```python
class QALoraLinearVariant(LoraVariant):
    @staticmethod
    def init(module: Linear, adapter_name: str, **kwargs: Any) -> None:
        """
        Initialize QALoRA with grouped pooling.

        Args:
            qalora_group_size: Size of groups for pooling (required in kwargs)
        """
        if "qalora_group_size" not in kwargs:
            raise ValueError("use_qalora=True requires qalora_group_size in kwargs")

        if module.in_features % kwargs["qalora_group_size"] != 0:
            raise ValueError(
                f"in_features ({module.in_features}) must be divisible by "
                f"qalora_group_size ({kwargs['qalora_group_size']})"
            )

        qalora_group_size = kwargs["qalora_group_size"]

        # Add to parameter names
        if "qalora_group_size" not in module.other_param_names:
            module.other_param_names = module.other_param_names + ("qalora_group_size",)

        if not hasattr(module, "qalora_group_size"):
            module.qalora_group_size = {}
        module.qalora_group_size[adapter_name] = qalora_group_size

        # Resize lora_A for pooled input
        old_lora_A_layer = module.lora_A[adapter_name]
        r = old_lora_A_layer.out_features
        device = old_lora_A_layer.weight.device
        dtype = old_lora_A_layer.weight.dtype

        new_lora_A_layer = nn.Linear(
            old_lora_A_layer.in_features // qalora_group_size,
            r,
            bias=False,
            device=device,
            dtype=dtype,
        )
        module.lora_A[adapter_name] = new_lora_A_layer

    @staticmethod
    def forward(
        module: Linear,
        active_adapter: str,
        x: torch.Tensor,
        result: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward with grouped pooling"""
        lora_A_weight = module.lora_A[active_adapter].weight
        lora_B_weight = module.lora_B[active_adapter].weight
        dropout = module.lora_dropout[active_adapter]
        scaling = module.scaling[active_adapter]
        group_size = module.qalora_group_size[active_adapter]

        # Apply dropout
        x_dropped = dropout(x) if module.training and not isinstance(dropout, nn.Identity) else x
        orig_shape = x_dropped.shape

        # Reshape to 2D
        if len(orig_shape) > 2:
            x_flat = x_dropped.view(-1, module.in_features)
        else:
            x_flat = x_dropped

        batch_size, in_features = x_flat.shape
        pooled_features = in_features // group_size

        # Group and average pool
        x_pooled = x_flat.view(batch_size, pooled_features, group_size).mean(dim=2)

        # Scale to maintain magnitude
        x_pooled_scaled = x_pooled * pooled_features

        # LoRA computation
        delta = x_pooled_scaled @ lora_A_weight.t() @ lora_B_weight.t() * scaling

        # Reshape back
        if len(orig_shape) > 2:
            delta = delta.view(orig_shape[:-1] + (delta.size(-1),))

        return result + delta

    @staticmethod
    def merge_safe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("QALoRA for GPTQ layers does not support safe_merge")

    @staticmethod
    def merge_unsafe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> None:
        raise NotImplementedError("QALoRA for GPTQ layers does not support merge_unsafe")

    @staticmethod
    def unmerge(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("QALoRA for GPTQ layers does not support unmerge")

    @staticmethod
    def get_delta_weight(module: Linear, active_adapter: str) -> torch.Tensor:
        raise NotImplementedError("QALoRA for GPTQ layers does not support get_delta_weight")
```

### aLoRA (Activated LoRA)

aLoRA activates only for tokens following specific invocation sequences:

```python
class ALoraLinearVariant(LoraVariant):
    @staticmethod
    def init(module: Linear, adapter_name: str, **kwargs: Any) -> None:
        """No special initialization needed"""
        pass

    @staticmethod
    def forward(
        module: Linear,
        active_adapter: str,
        x: torch.Tensor,
        result: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward with token-selective activation.

        Uses alora_offsets from kwargs to determine which tokens to activate.
        """
        alora_offsets = kwargs.get("alora_offsets", None)
        lora_A = module.lora_A[active_adapter]
        lora_B = module.lora_B[active_adapter]
        dropout = module.lora_dropout[active_adapter]
        scaling = module.scaling[active_adapter]

        x = x.to(lora_A.weight.dtype)
        result_shape = result.shape
        B = result_shape[0]  # batch
        T = result_shape[1] if len(result_shape) == 3 else 1  # tokens
        D = result_shape[-1]  # dimensions
        Dx = x.shape[-1]
        device = result.device

        # Create activation mask
        if alora_offsets is None:
            mask = torch.zeros((B, T), dtype=torch.bool)
        else:
            # Convert None -> 0 and clip to T
            offsets = torch.tensor(
                [0 if o is None else min(int(o), T) for o in alora_offsets],
                device=device,
                dtype=torch.long,
            )

            # Mask True on last offsets[i] positions for each row
            pos = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
            mask = pos >= (T - offsets).unsqueeze(1)

        # Flatten for vectorization
        x_flat = x.view(-1, Dx)
        res_flat = result.view(-1, D)
        mask_flat = mask.view(-1)

        # Apply adapter only to selected tokens
        res_flat[mask_flat] += lora_B(lora_A(dropout(x_flat[mask_flat]))) * scaling

        return result

    @staticmethod
    def merge_safe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("aLoRA does not support safe merging")

    @staticmethod
    def merge_unsafe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> None:
        raise NotImplementedError("aLoRA does not support merging")

    @staticmethod
    def unmerge(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("aLoRA does not support unmerging")
```

**Helper functions for aLoRA:**

```python
def calculate_alora_offsets(
    peft_config: PeftConfig,
    active_adapter: str,
    input_ids: torch.Tensor,
    adapter_names: Optional[list[str]] = None
) -> list[int]:
    """
    Search each input sequence for last occurrence of invocation tokens.

    Returns list of offsets (distance from end) for each sequence.
    """
    if input_ids is None:
        return []

    batch_size = input_ids.shape[0]
    alora_offsets = [None] * batch_size
    cached_invocation_tensors = {}
    adapters_to_process_indices = collections.defaultdict(list)

    # Group sequences by adapter
    for i in range(batch_size):
        current_adapter_name = adapter_names[i] if adapter_names and i < len(adapter_names) else active_adapter

        if current_adapter_name == "__base__":
            alora_offsets[i] = None
            continue

        if current_adapter_name not in peft_config:
            warnings.warn(f"Adapter '{current_adapter_name}' not found. Using base model for row {i}.")
            alora_offsets[i] = None
            continue

        current_peft_config = peft_config[current_adapter_name]
        invocation_tokens = getattr(current_peft_config, "alora_invocation_tokens", None)

        if invocation_tokens is None:
            alora_offsets[i] = None
            continue

        if current_adapter_name not in cached_invocation_tensors:
            cached_invocation_tensors[current_adapter_name] = torch.tensor(
                invocation_tokens, dtype=torch.long, device=input_ids.device
            )

        adapters_to_process_indices[current_adapter_name].append(i)

    # Find invocation sequences
    for adapter_name_to_process, indices in adapters_to_process_indices.items():
        current_invocation_ids_tensor = cached_invocation_tensors[adapter_name_to_process]
        invocation_len = len(current_invocation_ids_tensor)

        for i in indices:
            sequence = input_ids[i]
            seq_len = len(sequence)
            best_match_start_idx = -1

            # Find all possible starts
            possible_starts = (sequence == current_invocation_ids_tensor[0]).nonzero(as_tuple=True)[0]

            # Check each start position
            for start_idx_tensor in possible_starts:
                idx = start_idx_tensor.item()
                if idx + invocation_len <= seq_len:
                    if torch.equal(sequence[idx : idx + invocation_len], current_invocation_ids_tensor):
                        if idx > best_match_start_idx:
                            best_match_start_idx = idx

            # Compute offset from end
            if best_match_start_idx != -1:
                offset_val = seq_len - best_match_start_idx
                alora_offsets[i] = offset_val if offset_val > 0 else None
            else:
                alora_offsets[i] = None

    return alora_offsets


def get_alora_offsets_for_forward(
    model: nn.Module,
    input_ids: torch.Tensor = None,
    inputs_embeds: torch.Tensor = None,
    **kwargs
):
    """Wrapper for calculate_alora_offsets in model.forward()"""
    adapter_names_for_offset_calc = kwargs.get("adapter_names", None)

    if not is_alora_relevant_in_batch(model, adapter_names_for_offset_calc):
        return kwargs

    alora_offsets = kwargs.get("alora_offsets")
    if alora_offsets is None:
        if input_ids is None and inputs_embeds is not None:
            warnings.warn("Cannot calculate aLoRA offsets with only inputs_embeds. Disabling aLoRA.")
            kwargs["alora_offsets"] = None
        elif input_ids is not None:
            kwargs["alora_offsets"] = calculate_alora_offsets(
                model.peft_config,
                model.active_adapter,
                input_ids,
                adapter_names=adapter_names_for_offset_calc,
            )
        else:
            kwargs["alora_offsets"] = None

    return kwargs
```

### BdLoRA (Block-Diagonal LoRA)

BdLoRA uses block-diagonal structure for efficient computation:

```python
class BlockDiagonalLinear(nn.Module):
    """
    Block-diagonal linear layer for structured sparsity.

    Implements: y = Wx where W has block-diagonal structure.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        nblocks: int,
        init_zero: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nblocks = nblocks

        if in_features % nblocks != 0 or out_features % nblocks != 0:
            raise ValueError(
                f"in_features={in_features} or out_features={out_features} "
                f"not divisible by {nblocks}"
            )

        # Weight shape: (out_features, in_features // nblocks)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features // nblocks, dtype=dtype, device=device)
        )

        if init_zero:
            torch.nn.init.zeros_(self.weight)
        else:
            torch.nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Block-diagonal matrix multiplication"""
        first_dims = x.shape[:-1]
        if x.dim() != 2:
            x = x.reshape(-1, x.shape[-1])

        B = x.shape[0]
        nb = self.nblocks
        m = x.shape[-1] // nb  # input per block
        n = self.out_features // nb  # output per block

        # Reshape for block computation
        x = x.reshape(B, nb, m)
        w = self.weight.view(nb, n, m)

        # Einstein summation: batch × blocks × (features per block)
        out = torch.einsum("bim,inm->bin", x, w)

        return out.reshape(*first_dims, -1)

    def weight_as_blockdiagonal_matrix(self):
        """Returns weight in standard matrix format with off-diagonal zeros"""
        return torch.block_diag(*torch.chunk(self.weight, self.nblocks, dim=0))


class BdLoraLinearVariant(LoraVariant):
    @staticmethod
    def init(module: Linear, adapter_name: str, **kwargs) -> None:
        """Initialize block-diagonal LoRA matrices"""
        use_bdlora = kwargs.get("use_bdlora")
        target_name = kwargs.get("target_name", "")

        # Handle dict config (from saved state)
        if isinstance(use_bdlora, dict):
            use_bdlora = BdLoraConfig(**use_bdlora)

        lora_a_blockdiagonal_pattern = use_bdlora.target_modules_bd_a or []
        lora_b_blockdiagonal_pattern = use_bdlora.target_modules_bd_b or []
        nblocks = use_bdlora.nblocks

        # Check which matrix to make block-diagonal
        has_lora_a_blockdiagonal = any(pattern in target_name for pattern in lora_a_blockdiagonal_pattern)
        has_lora_b_blockdiagonal = any(pattern in target_name for pattern in lora_b_blockdiagonal_pattern)

        if has_lora_a_blockdiagonal and has_lora_b_blockdiagonal:
            raise ValueError(f"Target {target_name} matches both A and B patterns")

        if use_bdlora.match_strict and not (has_lora_a_blockdiagonal or has_lora_b_blockdiagonal):
            raise ValueError(
                f"Target {target_name} matches neither A nor B patterns. "
                "Set match_strict=False if intentional."
            )

        # Replace lora_A with block-diagonal version
        if has_lora_a_blockdiagonal:
            r = module.lora_A[adapter_name].out_features
            base_layer = module.get_base_layer()
            layer = BlockDiagonalLinear(
                base_layer.in_features,
                r,
                nblocks=nblocks,
                init_zero=False,
                dtype=base_layer.weight.dtype,
                device=base_layer.weight.device,
            )
            module.lora_A[adapter_name] = layer

        # Replace lora_B with block-diagonal version
        elif has_lora_b_blockdiagonal:
            r = module.lora_B[adapter_name].in_features
            base_layer = module.get_base_layer()
            layer = BlockDiagonalLinear(
                r,
                base_layer.out_features,
                nblocks=nblocks,
                init_zero=True,
                dtype=base_layer.weight.dtype,
                device=base_layer.weight.device,
            )
            module.lora_B[adapter_name] = layer

    @staticmethod
    def forward(
        module: Linear,
        active_adapter: str,
        x: torch.Tensor,
        result: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Forward with block-diagonal matrices"""
        lora_A = module.lora_A[active_adapter]
        lora_B = module.lora_B[active_adapter]
        dropout = module.lora_dropout[active_adapter]
        scaling = module.scaling[active_adapter]

        x = dropout(x)
        x = module._cast_input_dtype(x, lora_A.weight.dtype)
        result += lora_B(lora_A(x)) * scaling
        return result

    @staticmethod
    def _get_weight_from_module_maybe_blockdiagonal(module: nn.Module) -> torch.Tensor:
        """Extract weight matrix (convert block-diagonal to full if needed)"""
        if isinstance(module, BlockDiagonalLinear):
            return module.weight_as_blockdiagonal_matrix()
        else:
            return module.weight

    @staticmethod
    def _get_bdlora_delta_weight(module: Linear, adapter: str) -> torch.Tensor:
        """Compute delta weight for block-diagonal LoRA"""
        device = module.lora_B[adapter].weight.device
        base_layer = module.get_base_layer()
        dtype = base_layer.weight.dtype

        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = BdLoraLinearVariant._get_weight_from_module_maybe_blockdiagonal(module.lora_A[adapter])
        weight_B = BdLoraLinearVariant._get_weight_from_module_maybe_blockdiagonal(module.lora_B[adapter])

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, module.fan_in_fan_out) * module.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

        return output_tensor.to(dtype=dtype)

    @staticmethod
    def merge_safe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        return orig_weight + BdLoraLinearVariant._get_bdlora_delta_weight(module, active_adapter)

    @staticmethod
    def merge_unsafe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> None:
        orig_weight.data += BdLoraLinearVariant._get_bdlora_delta_weight(module, active_adapter)

    @staticmethod
    def unmerge(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        return orig_weight - BdLoraLinearVariant._get_bdlora_delta_weight(module, active_adapter)
```

## I/O Contract

### Common Variant Interface

All variants implement the `LoraVariant` protocol:

**init(module, adapter_name, **kwargs)**
- **Inputs**: LoRA module, adapter name, variant-specific kwargs
- **Outputs**: None (modifies module in-place)
- **Purpose**: Initialize variant-specific components

**forward(module, active_adapter, x, result, **kwargs)**
- **Inputs**: Module, adapter name, input tensor, base result, variant kwargs
- **Outputs**: Final result tensor
- **Purpose**: Compute variant-specific forward pass

**merge_safe(module, active_adapter, orig_weight)**
- **Inputs**: Module, adapter name, original weight
- **Outputs**: New merged weight tensor
- **Purpose**: Safe merge (no in-place modification)

**merge_unsafe(module, active_adapter, orig_weight)**
- **Inputs**: Module, adapter name, original weight
- **Outputs**: None (in-place modification)
- **Purpose**: Efficient in-place merge

**unmerge(module, active_adapter, orig_weight)**
- **Inputs**: Module, adapter name, merged weight
- **Outputs**: Unmerged weight tensor
- **Purpose**: Reverse merge operation

## Usage Examples

See individual variant documentation for detailed examples:
- DoRA: `huggingface_peft_DoraLayers.md`
- Arrow: `huggingface_peft_ArrowLinearVariant.md`
- QALoRA, aLoRA, BdLoRA: Examples below

### QALoRA Usage

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    use_qalora=True,
    qalora_group_size=32,  # Pool every 32 input features
)

peft_model = get_peft_model(quantized_model, config)
# Works best with GPTQ/AWQ quantized models
```

### aLoRA Usage

```python
from peft import LoraConfig, get_peft_model

# Define invocation tokens (e.g., special task prefix)
invocation_tokens = [50256, 50257]  # Example token IDs

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    alora_invocation_tokens=invocation_tokens,
)

peft_model = get_peft_model(model, config)

# During inference, adapter activates only after invocation sequence
inputs = tokenizer("Task: summarize\n\nText to summarize...", return_tensors="pt")
outputs = peft_model(**inputs)
```

### BdLoRA Usage

```python
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.config import BdLoraConfig

bd_config = BdLoraConfig(
    nblocks=4,  # 4 diagonal blocks
    target_modules_bd_a=["q_proj"],  # Block-diagonal for lora_A
    target_modules_bd_b=["v_proj"],  # Block-diagonal for lora_B
    match_strict=True,
)

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    use_bdlora=bd_config,
)

peft_model = get_peft_model(model, config)
# Reduces parameters by factor of nblocks for selected modules
```

## Related Pages

### Individual Variant Documentation
- `huggingface_peft_DoraLayers.md` - DoRA layer implementations
- `huggingface_peft_ArrowLinearVariant.md` - Arrow MoE routing

### Base Components
- `huggingface_peft_LoraVariant.md` - Base variant interface
- `huggingface_peft_LoraLayer.md` - Base LoRA layer
- `huggingface_peft_Linear.md` - Standard LoRA linear

### Configuration
- `huggingface_peft_LoraConfig.md` - Configuration with variant options
- `huggingface_peft_BdLoraConfig.md` - Block-diagonal configuration

### Concepts
- LoRA variant design patterns
- Weight decomposition techniques
- Structured sparsity in neural networks
- Token-selective activation strategies
- Quantization-aware training
