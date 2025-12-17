---
library: huggingface_peft
module: src/peft/tuners/lora/variants.py
classes: ["ArrowLinearVariant", "ArrowLoraLinearLayer"]
type: implementation
tags: ["lora", "arrow", "moe", "routing", "mixture-of-experts", "adapter"]
description: Arrow (Adaptive Routing for LoRa and Other Weight-based experts) - MoE-based LoRA routing
version: c6db49996c63
language: en
---

# ArrowLinearVariant and ArrowLoraLinearLayer

## Overview

The ArrowLinearVariant implements Arrow (Adaptive Routing for LoRa and Other Weight-based experts), a Mixture-of-Experts (MoE) approach for dynamically routing tokens through multiple LoRA adapters. Unlike traditional LoRA which applies fixed adapters, Arrow computes cosine similarity between input tokens and adapter prototypes to select and weight the top-k most relevant experts per token.

Key features:
- **Dynamic Token-Level Routing**: Each token is routed to its top-k most relevant LoRA experts
- **Prototype-Based Similarity**: Uses SVD-derived prototypes for efficient expert selection
- **General Knowledge Subtraction (GenKnowSub)**: Optional technique to purify task-specific adapters
- **Non-Mergeable**: Due to dynamic routing, adapters cannot be merged into base weights

The implementation consists of two main classes:
- `ArrowLinearVariant`: Integration with PEFT's LoRA variant system
- `ArrowLoraLinearLayer`: Core routing logic and forward pass computation

**Source File:** `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/lora/variants.py` (lines 33-131)
**Related File:** `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/lora/arrow.py` (476 lines)

## Code Reference

### ArrowLinearVariant Class

The variant adapter that integrates Arrow routing with PEFT's LoRA system:

```python
class ArrowLinearVariant(LoraVariant):
    @staticmethod
    def init(module: Linear, adapter_name: str, **kwargs):
        """Initialize ArrowLoraLinearLayer inside lora_arrow ModuleDict"""
        arrow_config = kwargs.get("arrow_config")
        if arrow_config is None:
            raise ValueError("ArrowLinearVariant.init() did not receive an arrow_config")

        # Build the ArrowLoRALayer
        arrow_layer = ArrowLoraLinearLayer(
            in_features=module.in_features,
            arrow_config=arrow_config,
        ).to(module.weight.device)

        # Register container if it doesn't exist
        if not hasattr(module, "lora_arrow"):
            module.lora_arrow = nn.ModuleDict()

        module.lora_arrow[adapter_name] = arrow_layer
```

Key characteristics:
- Creates `ArrowLoraLinearLayer` and stores it in `module.lora_arrow` ModuleDict
- Requires `arrow_config` containing routing parameters
- Device placement matches the module's weight device

### Arrow Forward Pass

```python
@staticmethod
def forward(
    module: Linear,
    *,
    active_adapter: str,
    x: torch.Tensor,
    result: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Arrow routing forward pass"""
    arrow = module.lora_arrow[active_adapter]

    # Apply GenKnowSub on first pass if applicable
    arrow.gen_know_sub(module.lora_A, module.lora_B)

    # Build prototypes on first pass after GenKnowSub
    arrow.build_prototypes(module.lora_A, module.lora_B)

    # Perform Arrow routing
    delta = arrow(
        x,
        lora_A=module.lora_A,
        lora_B=module.lora_B,
        dropout=module.lora_dropout[active_adapter],
        scaling=module.scaling,
    )
    return result + delta
```

Features:
- Lazy initialization of GenKnowSub and prototypes on first forward pass
- Routing performed by `ArrowLoraLinearLayer.__call__`
- Delta weights computed per-token based on expert selection

### ArrowLoraLinearLayer Class

Core routing implementation with prototype-based expert selection:

```python
class ArrowLoraLinearLayer(nn.Module):
    def __init__(self, in_features, arrow_config):
        super().__init__()
        self.in_features = in_features
        self._protos_ready = False
        self.top_k = arrow_config.top_k
        self.temperature = arrow_config.router_temperature
        self.rng_seed = arrow_config.rng_seed
        self.task_adapter_names = arrow_config.task_adapter_names.copy()
        self.gks_adapter_names = arrow_config.gks_adapter_names
        self.use_gks = arrow_config.use_gks
        self.gks_done = False
```

**Prototype Computation** (Right Singular Vectors):

```python
def top_right_singular_vec_from_BA(self, A, B, iters=15, eps=1e-8):
    """
    Computes top right singular vector of ΔW = B @ A without forming ΔW.

    Theory: For ΔW = B @ A, the right singular vectors are eigenvectors
    of ΔWᵀΔW = Aᵀ(BᵀB)A. Uses power iteration to find dominant eigenvector.
    """
    A32 = A.to(torch.float32)
    B32 = B.to(torch.float32)
    C = B32.T @ B32  # (r, r)

    # Initialize random vector
    gen = torch.Generator(device=A32.device.type) if self.rng_seed else None
    if gen:
        gen.manual_seed(int(self.rng_seed))
    v = torch.randn(A32.size(1), dtype=A32.dtype, device=A32.device, generator=gen)
    v = v / (v.norm() + eps)

    # Power iteration
    for _ in range(iters):
        w = A32.T @ (C @ (A32 @ v))
        v = w / (w.norm() + eps)

    return v  # fp32
```

**Routing Forward Pass**:

```python
def forward(self, x, lora_A, lora_B, dropout, scaling):
    """Apply Arrow routing logic"""
    x = self._cast_input_dtype(x, lora_A[self.task_adapter_names[0]].weight.dtype)
    B, *rest, F_in = x.shape
    tok = x.view(-1, F_in)  # Flatten to (t, F_in)
    t, E = tok.size(0), self.prototypes.size(0)

    # Convert scaling dict to tensor
    scales_tens = torch.tensor(
        [scaling[n] for n in self.task_adapter_names],
        device=tok.device, dtype=tok.dtype
    )  # (E,)

    # 1) Compute similarity (sign-agnostic)
    sim = torch.abs(tok @ self.prototypes.T)  # (t, E)

    # 2) Top-k selection + softmax over all experts
    top_v, idx = torch.topk(sim, self.top_k, dim=1)
    full_score = tok.new_full((t, E), float("-inf"))
    full_score.scatter_(1, idx, top_v)
    coeff = torch.softmax(full_score / self.temperature, dim=1)  # (t, E)

    # 3) Stack all A and B weights
    A_stack = torch.stack([lora_A[n].weight for n in self.task_adapter_names], dim=0)
    B_stack = torch.stack([lora_B[n].weight for n in self.task_adapter_names], dim=0)

    # 4) Project tokens into each expert's low-rank space
    z = torch.einsum("tf, erf -> ter", tok, A_stack)

    # 5) Lift back to output space
    y = torch.einsum("ter, eor -> teo", z, B_stack)

    # 6) Apply per-expert scaling
    y = y * scales_tens.view(1, -1, 1)

    # 7) Weighted sum over experts
    delta_flat = torch.einsum("te, teo -> to", coeff, y)

    # 8) Apply dropout and reshape
    delta = dropout(delta_flat)
    out_dim = delta_flat.size(-1)
    return delta.view(B, *rest, out_dim)
```

### General Knowledge Subtraction (GenKnowSub)

```python
@torch.no_grad()
def gen_know_sub(self, lora_A, lora_B):
    """
    Perform General Knowledge Subtraction to purify task adapters.
    Implements "forgetting-via-negation" principle from task arithmetic.
    """
    if not self.use_gks:
        return
    elif self.gks_done and not self.gks_added_adapter_names:
        return

    # Compute average A/B over gks_adapter_names
    avg_A = torch.stack([lora_A[n].weight for n in self.gks_adapter_names], dim=0).mean(0)
    avg_B = torch.stack([lora_B[n].weight for n in self.gks_adapter_names], dim=0).mean(0)

    # Subtract from task-specific experts
    if not self.gks_done:  # First time - all experts
        for name in self.task_adapter_names:
            lora_A[name].weight.data.sub_(avg_A)
            lora_B[name].weight.data.sub_(avg_B)
    else:  # Only newly added experts
        for name in self.gks_added_adapter_names:
            lora_A[name].weight.data.sub_(avg_A)
            lora_B[name].weight.data.sub_(avg_B)

    self.gks_done = True
    self.gks_added_adapter_names = []
```

### Non-Mergeable Nature

```python
@staticmethod
def merge_safe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
    raise RuntimeError("Cannot merge an active Arrow router adapter. Remove it first.")

@staticmethod
def merge_unsafe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> None:
    raise RuntimeError("Cannot merge an active Arrow router adapter. Remove it first.")

@staticmethod
def unmerge(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
    raise RuntimeError("Cannot unmerge an active Arrow router adapter. Remove it first.")
```

Arrow routing performs per-token dynamic expert selection, making it impossible to represent as a single merged weight matrix.

## I/O Contract

### ArrowLinearVariant.init()

**Inputs:**
- `module` (Linear): LoRA layer containing base_layer, lora_A, lora_B
- `adapter_name` (str): Name of the adapter (default: "arrow_router")
- `arrow_config` (ArrowConfig): Configuration object with routing parameters

**Outputs:**
- None (modifies module in-place by adding `lora_arrow` ModuleDict)

**Side Effects:**
- Creates `ArrowLoraLinearLayer` and registers it in `module.lora_arrow[adapter_name]`
- Device placement matches module.weight.device

### ArrowLinearVariant.forward()

**Inputs:**
- `module` (Linear): LoRA layer with Arrow routing
- `active_adapter` (str): Name of active adapter
- `x` (torch.Tensor): Input tensor, shape `(batch, ..., in_features)`
- `result` (torch.Tensor): Base layer output
- `**kwargs`: Additional arguments (ignored for compatibility)

**Outputs:**
- `torch.Tensor`: `result + delta`, where delta is routed LoRA contribution

**Side Effects:**
- First call: Applies GenKnowSub and builds prototypes (lazy initialization)
- Modifies adapter weights in-place if GenKnowSub is active

### ArrowLoraLinearLayer.forward()

**Inputs:**
- `x` (torch.Tensor): Input tensor, shape `(batch, seq_len, in_features)` or `(batch, in_features)`
- `lora_A` (ModuleDict): Dictionary of LoRA A matrices
- `lora_B` (ModuleDict): Dictionary of LoRA B matrices
- `dropout` (nn.Module): Dropout layer for current adapter
- `scaling` (dict): Dictionary mapping adapter names to scaling factors

**Outputs:**
- `torch.Tensor`: Weighted combination of expert outputs, shape matches input except last dimension

**Computational Complexity:**
- Similarity computation: O(t × E × F_in)
- Top-k selection: O(t × E × log(k))
- Expert projection: O(t × E × r × F_in) + O(t × E × r × F_out)
- Where t=tokens, E=experts, r=rank, F_in/F_out=input/output features

## Usage Examples

### Creating an Arrow Model

```python
from peft import PeftModel
from peft.tuners.lora.arrow import create_arrow_model, ArrowConfig
from transformers import AutoModelForCausalLM

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Define Arrow configuration
arrow_config = ArrowConfig(
    top_k=2,                    # Select top-2 experts per token
    router_temperature=0.1,      # Temperature for softmax routing
    use_gks=True,               # Enable General Knowledge Subtraction
    task_adapter_names=[],      # Populated automatically
    gks_adapter_names=[],       # Populated automatically
    rng_seed=42                 # For reproducible prototype initialization
)

# Paths to task-specific LoRA adapters
task_adapters = [
    "path/to/adapter1",
    "path/to/adapter2",
    "path/to/adapter3",
]

# Optional: paths to general knowledge adapters for GenKnowSub
general_adapters = [
    "path/to/general_adapter1",
    "path/to/general_adapter2",
]

# Create Arrow model
model = create_arrow_model(
    base_model=base_model,
    task_specific_adapter_paths=task_adapters,
    arrow_config=arrow_config,
    general_adapter_paths=general_adapters,  # Only if use_gks=True
)
```

### Inference with Arrow Routing

```python
import torch

# Prepare input
text = "The capital of France is"
inputs = tokenizer(text, return_tensors="pt")

# Forward pass with Arrow routing
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Each token is routed to top-k experts dynamically
predictions = torch.argmax(logits, dim=-1)
```

### Understanding Arrow's Architecture

```python
# Inspect Arrow layer structure
for name, module in model.named_modules():
    if hasattr(module, 'lora_arrow'):
        print(f"Layer: {name}")
        arrow_layer = module.lora_arrow['arrow_router']
        print(f"  - Number of experts: {len(arrow_layer.task_adapter_names)}")
        print(f"  - Top-k: {arrow_layer.top_k}")
        print(f"  - Temperature: {arrow_layer.temperature}")
        print(f"  - Prototypes ready: {arrow_layer._protos_ready}")
        if hasattr(arrow_layer, 'prototypes'):
            print(f"  - Prototype shape: {arrow_layer.prototypes.shape}")
```

### Dynamic Adapter Management

```python
# Arrow supports adding new adapters after creation
model.load_adapter("path/to/new_adapter", adapter_name="task_3")

# Arrow automatically detects new adapters and updates routing
# on_adapter_change() is called to refresh prototypes
for name, module in model.named_modules():
    if hasattr(module, 'lora_arrow'):
        arrow_layer = module.lora_arrow['arrow_router']
        arrow_layer.on_adapter_change(module.lora_A, module.lora_B)
```

### Without General Knowledge Subtraction

```python
# Arrow can work without GenKnowSub
arrow_config = ArrowConfig(
    top_k=3,
    router_temperature=0.1,
    use_gks=False,  # Disable GenKnowSub
)

model = create_arrow_model(
    base_model=base_model,
    task_specific_adapter_paths=task_adapters,
    arrow_config=arrow_config,
    # No general_adapter_paths needed
)
```

### Routing Analysis

```python
# Hook to analyze routing decisions
routing_stats = []

def routing_hook(module, input, output):
    if hasattr(module, 'lora_arrow'):
        arrow = module.lora_arrow['arrow_router']
        # Access routing information from forward pass
        routing_stats.append({
            'layer': module.__class__.__name__,
            'n_experts': len(arrow.task_adapter_names),
            'top_k': arrow.top_k,
        })

# Register hooks
hooks = []
for module in model.modules():
    if hasattr(module, 'lora_arrow'):
        hooks.append(module.register_forward_hook(routing_hook))

# Run inference
outputs = model(**inputs)

# Clean up hooks
for hook in hooks:
    hook.remove()

# Analyze routing
print(f"Total layers with Arrow routing: {len(routing_stats)}")
for stat in routing_stats:
    print(f"{stat['layer']}: {stat['n_experts']} experts, top-{stat['top_k']}")
```

## Related Pages

### Core LoRA Components
- `huggingface_peft_LoraVariant.md` - Base class for all LoRA variants
- `huggingface_peft_Linear.md` - Base LoRA linear layer implementation
- `huggingface_peft_LoraLayer.md` - LoRA layer base class

### Other LoRA Variants
- `huggingface_peft_DoraLinearVariant.md` - Weight-decomposed LoRA with magnitude and direction
- `huggingface_peft_QALoraLinearVariant.md` - Quantization-aware LoRA
- `huggingface_peft_ALoraLinearVariant.md` - Activated LoRA with invocation tokens

### Configuration
- `huggingface_peft_ArrowConfig.md` - Arrow configuration dataclass
- `huggingface_peft_LoraConfig.md` - Base LoRA configuration

### Related Utilities
- `huggingface_peft_create_arrow_model.md` - Factory function for creating Arrow models
- `huggingface_peft_check_loaded_lora_compatibility_arrow.md` - Adapter compatibility validation

### Concepts
- Mixture-of-Experts (MoE) routing strategies
- Task arithmetic and adapter composition
- Prototype-based similarity for expert selection
- General Knowledge Subtraction for adapter purification
