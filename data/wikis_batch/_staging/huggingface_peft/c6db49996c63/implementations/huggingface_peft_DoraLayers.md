---
library: huggingface_peft
module: src/peft/tuners/lora/dora.py
classes: ["DoraLinearLayer", "DoraEmbeddingLayer", "DoraConv1dLayer", "DoraConv2dLayer", "DoraConv3dLayer"]
type: implementation
tags: ["lora", "dora", "weight-decomposition", "magnitude", "direction", "normalization"]
description: DoRA (Weight-Decomposed Low-Rank Adaptation) - decomposing weights into magnitude and direction
version: c6db49996c63
language: en
---

# DoRA: Weight-Decomposed Low-Rank Adaptation

## Overview

DoRA (Weight-Decomposed Low-Rank Adaptation) is an enhancement to LoRA that explicitly decomposes weight updates into magnitude and direction components. By learning a separate magnitude vector while LoRA handles directional updates, DoRA achieves better performance and training stability compared to standard LoRA.

Key features:
- **Magnitude-Direction Decomposition**: Separates weight updates into `magnitude * direction`
- **Learned Magnitude Vector**: Trainable per-output-feature magnitude parameters
- **Enhanced Stability**: Better gradient flow and training dynamics
- **Column-wise Normalization**: Applies L2 normalization to weight columns
- **Multi-Layer Support**: Works with Linear, Embedding, Conv1d, Conv2d, Conv3d layers

The core idea from the DoRA paper (https://huggingface.co/papers/2402.09353):
```
W' = m * (W + ΔW) / ||W + ΔW||_c
```
where:
- `W` is the original weight
- `ΔW` is the LoRA update (lora_B @ lora_A)
- `m` is the learned magnitude vector
- `||·||_c` is column-wise L2 norm

**Source File:** `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/lora/dora.py` (203 lines)
**Reference Paper:** https://huggingface.co/papers/2402.09353

## Code Reference

### DoraLinearLayer Class

Core implementation for linear layers:

```python
class DoraLinearLayer(nn.Module):
    def __init__(self, fan_in_fan_out):
        super().__init__()
        self.fan_in_fan_out = fan_in_fan_out

    def get_weight_norm(self, weight, lora_weight, scaling) -> torch.Tensor:
        """
        Calculate L2 norm of weight matrix, column-wise.

        Args:
            weight: Base weight matrix
            lora_weight: LoRA weight delta
            scaling: LoRA scaling factor

        Returns:
            weight_norm: Column-wise L2 norms, shape (out_features,)
        """
        # Transpose if needed
        weight = transpose(weight, self.fan_in_fan_out)

        # Combined weight
        weight = weight + scaling * lora_weight

        # L2 norm per output feature (column)
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm

    def update_layer(self, *, base_layer, lora_A, lora_B, scaling, place_on_cpu=False) -> None:
        """
        Initialize magnitude vector based on current weights.

        Args:
            base_layer: Base layer (Linear, quantized Linear, etc.)
            lora_A: LoRA A matrix
            lora_B: LoRA B matrix
            scaling: LoRA scaling factor
            place_on_cpu: Whether to place magnitude on CPU
        """
        # Handle FP16 (convert to FP32 temporarily for stability)
        dtype_is_fp16 = lora_A.dtype == torch.float16
        if dtype_is_fp16:
            lora_A = lora_A.float()
            lora_B = lora_B.float()

        with gather_params_ctx(base_layer.parameters()):
            # Handle quantized layers (need to create copy for FSDP compatibility)
            if base_layer.__class__.__name__ == "Linear4bit":
                base_layer = deepcopy(base_layer)

            # Dequantize if needed
            weight = dequantize_module_weight(base_layer)

            # Compute LoRA weight
            if weight.data.ndim >= 3:  # Conv layers
                r = lora_A.shape[0]
                lora_weight = torch.mm(lora_B.view([-1, r]), lora_A.view([r, -1]))
                lora_weight = lora_weight.reshape(weight.shape)
            else:  # Linear layers
                lora_weight = lora_B @ lora_A

            # Convert back to FP16 if needed
            if dtype_is_fp16:
                lora_weight = lora_weight.half()

            # Compute initial weight norm
            weight_norm = self.get_weight_norm(weight.to(lora_A.device), lora_weight, scaling)

        # Place on CPU if requested (memory optimization)
        if place_on_cpu:
            weight_norm = weight_norm.to("cpu")

        # Create learnable magnitude parameter
        self.weight = nn.Parameter(weight_norm, requires_grad=True)

    def forward(self, x, *, lora_A, lora_B, scaling, base_layer, base_result=None):
        """
        Compute DoRA output: magnitude-scaled directional update.

        Args:
            x: Input tensor
            lora_A: LoRA A layer
            lora_B: LoRA B layer
            scaling: LoRA scaling factor
            base_layer: Base layer
            base_result: Pre-computed base layer output (optional)

        Returns:
            DoRA contribution to add to base output
        """
        # Compute LoRA weight using forward passes (FSDP-safe)
        # Don't use lora_B.weight @ lora_A.weight directly
        x_eye = torch.eye(lora_A.weight.shape[1], device=lora_A.weight.device, dtype=x.dtype)
        lora_weight = lora_B(lora_A(x_eye)).T

        magnitude = self.weight
        weight = dequantize_module_weight(base_layer)
        weight = weight.to(x.dtype)

        # Compute weight norm with current LoRA
        weight_norm = self.get_weight_norm(weight, lora_weight.detach(), scaling)

        # Section 4.3 of DoRA paper: detach weight_norm from gradient graph
        # "we suggest treating ||V +∆V||_c in Eq. (5) as a constant,
        # thereby detaching it from the gradient graph"
        weight_norm = weight_norm.detach()

        # Magnitude normalization scale
        mag_norm_scale = (magnitude / weight_norm).view(1, -1)

        # Apply LoRA
        lora_result = lora_B(lora_A(x))

        # Compute base result if not provided
        bias = None
        if base_result is not None:
            bias = base_layer.bias
            if bias is not None:
                base_result = base_result - bias
        else:
            base_result = F.linear(x, transpose(weight, self.fan_in_fan_out))

        # DoRA contribution
        result_dora = (mag_norm_scale - 1) * base_result + mag_norm_scale * lora_result * scaling

        return result_dora

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora.dora." + rep
```

### DoraEmbeddingLayer Class

DoRA for embedding layers:

```python
class DoraEmbeddingLayer(DoraLinearLayer):
    def forward(self, x, *, lora_A, lora_B, scaling, base_layer, embed_fn):
        """
        DoRA for embeddings.

        Args:
            x: Input indices
            lora_A: LoRA A weights (transposed from lora_embedding_A)
            lora_B: LoRA B weights (transposed from lora_embedding_B)
            scaling: Scaling factor
            base_layer: Base embedding layer
            embed_fn: Embedding function to use

        Returns:
            mag_norm_scale: Magnitude normalization factor
            result_dora: DoRA embedding contribution
        """
        # Compute LoRA weight
        lora_weight = (lora_A @ lora_B).T

        magnitude = self.weight
        weight = base_layer.weight

        # Get weight norm
        weight_norm = self.get_weight_norm(weight, lora_weight.detach(), scaling)

        # Detach per DoRA paper section 4.3
        weight_norm = weight_norm.detach()

        # Magnitude normalization
        mag_norm_scale = magnitude / weight_norm

        # DoRA embedding result
        result_dora = mag_norm_scale * (embed_fn(x, lora_A) @ lora_B) * scaling

        return mag_norm_scale, result_dora

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora.dora." + rep
```

### Convolutional DoRA Layers

Base class for convolutional layers:

```python
class _DoraConvNdLayer(DoraLinearLayer):
    def get_weight_norm(self, weight, lora_weight, scaling) -> torch.Tensor:
        """
        Weight norm for convolutional layers.

        Computes L2 norm across all dimensions except output channels.
        """
        # Combined weight
        weight = weight + scaling * lora_weight

        # Norm across spatial and input channel dims
        dim = tuple(range(1, weight.dim()))
        weight_norm = weight.norm(p=2, dim=dim, keepdim=True).transpose(1, 0)
        return weight_norm

    def forward(self, x, *, lora_A, lora_B, scaling, base_layer, base_result=None):
        """
        DoRA forward for convolutional layers.

        Args:
            x: Input tensor (N, C_in, *)
            lora_A: LoRA A conv layer
            lora_B: LoRA B conv layer
            scaling: Scaling factor
            base_layer: Base conv layer
            base_result: Pre-computed base output

        Returns:
            DoRA contribution
        """
        weight = base_layer.weight
        r = lora_A.weight.shape[0]

        # Compute LoRA weight by reshaping conv weights
        lora_weight = torch.mm(lora_B.weight.view([-1, r]), lora_A.weight.view([r, -1]))
        lora_weight = lora_weight.reshape(weight.shape)

        magnitude = self.weight

        # Compute weight norm
        weight_norm = self.get_weight_norm(weight, lora_weight.detach(), scaling)
        weight_norm = weight_norm.detach()

        # Magnitude normalization
        mag_norm_scale = magnitude / weight_norm

        # Compute base result if not provided
        if base_result is None:
            base_result = self.conv_fn(
                x,
                weight,
                bias=None,
                stride=base_layer.stride,
                padding=base_layer.padding,
                dilation=base_layer.dilation,
                groups=base_layer.groups,
            )
        else:
            bias = base_layer.bias
            if bias is not None:
                # Reshape bias for broadcasting
                bias_shape = (1, -1) + (1,) * (base_result.dim() - 2)
                base_result = base_result - bias.view(*bias_shape)

        # DoRA result
        result_dora = (mag_norm_scale - 1) * base_result + mag_norm_scale * lora_B(lora_A(x)) * scaling
        return result_dora

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora.dora." + rep
```

**Specific Conv Layers:**

```python
class DoraConv1dLayer(_DoraConvNdLayer):
    def __init__(self, fan_in_fan_out):
        super().__init__(fan_in_fan_out)
        self.conv_fn = F.conv1d


class DoraConv2dLayer(_DoraConvNdLayer):
    def __init__(self, fan_in_fan_out):
        super().__init__(fan_in_fan_out)
        self.conv_fn = F.conv2d


class DoraConv3dLayer(_DoraConvNdLayer):
    def __init__(self, fan_in_fan_out):
        super().__init__(fan_in_fan_out)
        self.conv_fn = F.conv3d
```

## I/O Contract

### DoraLinearLayer.update_layer()

**Inputs:**
- `base_layer`: Base layer (Linear, Linear4bit, etc.)
- `lora_A` (torch.Tensor): LoRA A weight matrix, shape `(r, in_features)`
- `lora_B` (torch.Tensor): LoRA B weight matrix, shape `(out_features, r)`
- `scaling` (float): LoRA scaling factor
- `place_on_cpu` (bool): Whether to place magnitude on CPU

**Outputs:**
- None (creates `self.weight` parameter)

**Side Effects:**
- Creates learnable magnitude vector parameter
- May dequantize base layer weights
- Temporarily converts FP16 to FP32 for stability

**Parameter Created:**
- `self.weight`: Shape `(out_features,)`, learnable magnitude vector

### DoraLinearLayer.forward()

**Inputs:**
- `x` (torch.Tensor): Input, shape `(batch, ..., in_features)`
- `lora_A` (nn.Module): LoRA A layer
- `lora_B` (nn.Module): LoRA B layer
- `scaling` (float): Scaling factor
- `base_layer`: Base layer module
- `base_result` (Optional[torch.Tensor]): Pre-computed base output

**Outputs:**
- `torch.Tensor`: DoRA contribution, shape `(batch, ..., out_features)`

**Computational Complexity:**
- Weight norm computation: O(out_features × in_features)
- LoRA forward: O(batch × seq × in_features × r) + O(batch × seq × r × out_features)
- Element-wise ops: O(batch × seq × out_features)

**Mathematical Operation:**
```python
mag_norm_scale = magnitude / ||W + scaling * (lora_B @ lora_A)||_columns
result_dora = (mag_norm_scale - 1) * base_result + mag_norm_scale * lora_B(lora_A(x)) * scaling
```

### DoraEmbeddingLayer.forward()

**Inputs:**
- `x` (torch.Tensor): Input indices, shape `(batch, seq_len)`
- `lora_A`, `lora_B` (torch.Tensor): LoRA weight matrices (not modules)
- `scaling` (float): Scaling factor
- `base_layer`: Base embedding layer
- `embed_fn` (callable): Embedding function

**Outputs:**
- `mag_norm_scale` (torch.Tensor): Magnitude normalization, shape `(vocab_size,)`
- `result_dora` (torch.Tensor): DoRA contribution, shape `(batch, seq_len, embed_dim)`

**Note:** Returns tuple unlike linear version

### _DoraConvNdLayer.forward()

**Inputs:**
- `x` (torch.Tensor): Input, shape `(N, C_in, *spatial_dims)`
- `lora_A`, `lora_B` (nn.Module): LoRA conv layers
- `scaling` (float): Scaling factor
- `base_layer`: Base conv layer
- `base_result` (Optional[torch.Tensor]): Pre-computed base output

**Outputs:**
- `torch.Tensor`: DoRA contribution, shape `(N, C_out, *output_spatial_dims)`

**Spatial Dimensions:**
- Conv1d: `(N, C_in, L)` → `(N, C_out, L_out)`
- Conv2d: `(N, C_in, H, W)` → `(N, C_out, H_out, W_out)`
- Conv3d: `(N, C_in, D, H, W)` → `(N, C_out, D_out, H_out, W_out)`

## Usage Examples

### Basic DoRA Configuration

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure LoRA with DoRA
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    use_dora=True,  # Enable DoRA!
)

# Create PEFT model with DoRA
peft_model = get_peft_model(model, config)

# Training proceeds normally, but with magnitude-direction decomposition
```

### Inspecting DoRA Components

```python
# Check magnitude vectors
for name, module in peft_model.named_modules():
    if hasattr(module, 'lora_magnitude_vector'):
        for adapter_name, dora_layer in module.lora_magnitude_vector.items():
            print(f"{name}.{adapter_name}:")
            print(f"  Magnitude shape: {dora_layer.weight.shape}")
            print(f"  Magnitude mean: {dora_layer.weight.mean().item():.4f}")
            print(f"  Magnitude std: {dora_layer.weight.std().item():.4f}")
            print(f"  Magnitude min/max: {dora_layer.weight.min().item():.4f} / {dora_layer.weight.max().item():.4f}")
```

### Comparing LoRA vs DoRA

```python
import torch
from transformers import AutoModelForCausalLM

# Test on same model
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# Standard LoRA
config_lora = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    use_dora=False,
)
model_lora = get_peft_model(base_model, config_lora)

# DoRA
config_dora = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    use_dora=True,
)
model_dora = get_peft_model(base_model, config_dora)

# Count parameters
lora_params = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
dora_params = sum(p.numel() for p in model_dora.parameters() if p.requires_grad)

print(f"LoRA trainable params: {lora_params:,}")
print(f"DoRA trainable params: {dora_params:,}")
print(f"DoRA overhead: {dora_params - lora_params:,} params")
# DoRA has additional out_features parameters per layer for magnitudes
```

### Memory-Efficient DoRA (CPU Offloading)

```python
# For very large models, offload magnitude vectors to CPU
# This is handled internally during initialization with ephemeral_gpu_offload

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    use_dora=True,
)

# When using with 8-bit/4-bit quantization + FSDP, magnitude vectors
# are automatically placed on CPU when appropriate
from peft import prepare_model_for_kbit_training

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,
)
model = prepare_model_for_kbit_training(model)
peft_model = get_peft_model(model, config)

# Magnitude vectors will be on CPU if needed
```

### DoRA with Different Layer Types

```python
# DoRA for various layer types
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",     # Linear -> DoraLinearLayer
        "embed_tokens",  # Embedding -> DoraEmbeddingLayer
    ],
    use_dora=True,
)

peft_model = get_peft_model(model, config)

# Check layer types
for name, module in peft_model.named_modules():
    if hasattr(module, 'lora_magnitude_vector'):
        for adapter_name, dora_layer in module.lora_magnitude_vector.items():
            print(f"{name}: {type(dora_layer).__name__}")
```

### Training with DoRA

```python
from transformers import Trainer, TrainingArguments

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    use_dora=True,
)

peft_model = get_peft_model(model, config)

training_args = TrainingArguments(
    output_dir="./dora_finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,  # DoRA may benefit from slightly higher LR
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

### Merging DoRA Adapters

```python
# DoRA adapters can be merged like standard LoRA
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    use_dora=True,
)

peft_model = get_peft_model(model, config)

# Train...

# Merge adapter into base model
peft_model = peft_model.merge_and_unload()

# Save merged model
peft_model.save_pretrained("./dora_merged")
```

### Gradient Monitoring for DoRA

```python
# Monitor gradients for magnitude vectors
def log_dora_gradients(model):
    mag_grads = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora_magnitude_vector'):
            for adapter_name, dora_layer in module.lora_magnitude_vector.items():
                if dora_layer.weight.grad is not None:
                    grad_norm = dora_layer.weight.grad.norm().item()
                    mag_grads[f"{name}.{adapter_name}"] = grad_norm
    return mag_grads

# During training
outputs = peft_model(**inputs)
loss = outputs.loss
loss.backward()

mag_grads = log_dora_gradients(peft_model)
for layer, grad_norm in mag_grads.items():
    print(f"{layer} magnitude gradient norm: {grad_norm:.6f}")

optimizer.step()
optimizer.zero_grad()
```

### DoRA with Quantization

```python
from transformers import BitsAndBytesConfig

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# Prepare for training
model = prepare_model_for_kbit_training(model)

# Add DoRA adapters
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    use_dora=True,
)

peft_model = get_peft_model(model, config)

# DoRA works with quantized base weights!
```

### Analyzing Weight Norms

```python
# Analyze how weight norms evolve during training
import matplotlib.pyplot as plt

norms_history = []

for epoch in range(num_epochs):
    for batch in dataloader:
        # Training step...

        # Log norms periodically
        if step % 100 == 0:
            norms = {}
            for name, module in peft_model.named_modules():
                if hasattr(module, 'lora_magnitude_vector'):
                    for adapter_name, dora_layer in module.lora_magnitude_vector.items():
                        # Get base + LoRA weight
                        base_weight = module.get_base_layer().weight
                        lora_weight = module.get_delta_weight(adapter_name)
                        combined = base_weight + lora_weight

                        # Compute column-wise norms
                        weight_norm = torch.linalg.norm(combined, dim=1)

                        norms[f"{name}.{adapter_name}"] = {
                            'magnitude_mean': dora_layer.weight.mean().item(),
                            'weight_norm_mean': weight_norm.mean().item(),
                            'ratio': (dora_layer.weight / weight_norm).mean().item(),
                        }

            norms_history.append(norms)

# Plot evolution
# ... visualization code ...
```

## Related Pages

### Core LoRA Components
- `huggingface_peft_DoraLinearVariant.md` - Variant integration for DoRA
- `huggingface_peft_DoraEmbeddingVariant.md` - Variant for embedding layers
- `huggingface_peft_DoraConvNdVariant.md` - Variants for convolutional layers
- `huggingface_peft_LoraLayer.md` - Base LoRA layer
- `huggingface_peft_Linear.md` - Standard LoRA linear layer

### Configuration
- `huggingface_peft_LoraConfig.md` - Configuration with use_dora parameter

### Other LoRA Variants
- `huggingface_peft_ArrowLinearVariant.md` - MoE adaptive routing
- `huggingface_peft_QALoraLinearVariant.md` - Quantization-aware LoRA
- `huggingface_peft_ALoraLinearVariant.md` - Activated LoRA

### Utilities
- `huggingface_peft_dequantize_module_weight.md` - Weight dequantization for quantized models
- `huggingface_peft_gather_params_ctx.md` - Context manager for distributed parameters
- `huggingface_peft_transpose.md` - Weight transposition utilities

### Concepts
- Weight decomposition: magnitude vs direction
- Column-wise normalization in neural networks
- Gradient detachment for training stability
- LoRA with enhanced expressiveness
