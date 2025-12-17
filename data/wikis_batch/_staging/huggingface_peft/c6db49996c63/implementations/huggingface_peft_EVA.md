---
library: huggingface_peft
module: src/peft/tuners/lora/eva.py
classes: ["get_eva_state_dict", "initialize_lora_eva_weights", "SVDHook", "HashHook"]
type: implementation
tags: ["lora", "eva", "svd", "initialization", "rank-distribution", "incremental-pca"]
description: EVA (Eigenvalue Activation-aware) initialization for LoRA adapters with data-driven rank distribution
version: c6db49996c63
language: en
---

# EVA: Eigenvalue Activation-aware Initialization

## Overview

EVA (Eigenvalue Activation-aware initialization) is an advanced initialization method for LoRA adapters that uses data-driven techniques to determine optimal rank distribution across layers. Unlike traditional LoRA initialization which uses random or fixed rank values, EVA computes Singular Value Decomposition (SVD) on layer activations to initialize LoRA weights based on the actual data distribution.

Key features:
- **Data-Driven Rank Distribution**: Automatically determines optimal rank per layer based on explained variance
- **Incremental SVD**: Uses incremental PCA to efficiently compute SVD components on streaming data
- **Convergence Detection**: Monitors component stability using cosine similarity
- **Distributed Training Support**: Can gather inputs across multiple GPUs for better statistics
- **Layer-Specific Customization**: Supports custom preprocessing functions per layer

The implementation consists of several key components:
- **SVD Computation**: Incremental SVD on layer activations with convergence tracking
- **Rank Distribution**: Allocates rank budget based on explained variance ratios
- **State Dict Management**: Saves/loads EVA initialization state for reuse

**Source File:** `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/lora/eva.py` (739 lines)

## Code Reference

### Main API Functions

**get_eva_state_dict**: Compute EVA initialization state

```python
@torch.no_grad()
def get_eva_state_dict(
    model: torch.nn.Module,
    dataloader: Iterable,
    peft_config: Optional[LoraConfig] = None,
    forward_fn: Optional[callable] = forward_fn_dict,
    prepare_model_inputs_fn: Optional[callable] = prepare_model_inputs_fn_language_modeling,
    prepare_layer_inputs_fn: Union[callable, dict[str, callable], None] = prepare_layer_inputs_fn_language_modeling,
    adapter_name: str = "default",
    gather_distributed_inputs: bool = True,
    show_progress_bar: bool = True,
) -> dict:
    """
    Compute the SVD for each layer in the model.

    Args:
        model: The model to compute SVD for (can be PeftModel or base model)
        dataloader: Dataloader to use for forward pass
        peft_config: LoRA configuration (required if model is not PeftModel)
        forward_fn: Function to run forward pass: forward_fn(model, inputs)
        prepare_model_inputs_fn: Function to prepare model inputs for hooks
        prepare_layer_inputs_fn: Function to prepare layer inputs for SVD
        adapter_name: Name of adapter to compute SVD for
        gather_distributed_inputs: Whether to gather inputs across ranks
        show_progress_bar: Whether to show progress bar

    Returns:
        eva_state_dict: Dictionary mapping layer names to SVD components (U matrices)
    """
```

**initialize_lora_eva_weights**: Apply EVA initialization to a model

```python
@torch.no_grad()
def initialize_lora_eva_weights(
    model: torch.nn.Module,
    dataloader: Optional[Iterable] = None,
    eva_state_dict: Optional[dict] = None,
    forward_fn: Optional[callable] = forward_fn_dict,
    prepare_model_inputs_fn: Optional[callable] = prepare_model_inputs_fn_language_modeling,
    prepare_layer_inputs_fn: Union[callable, dict[str, callable], None] = prepare_layer_inputs_fn_language_modeling,
    adapter_name: str = "default",
    gather_distributed_inputs: bool = True,
    show_progress_bar: bool = True,
):
    """
    Initialize LoRA weights using EVA method.

    Args:
        model: PeftModel to initialize
        dataloader: Dataloader (required if eva_state_dict not provided)
        eva_state_dict: Pre-computed state dict (if None, computed from dataloader)
        forward_fn: Function to run forward pass
        prepare_model_inputs_fn: Function to prepare model inputs
        prepare_layer_inputs_fn: Function to prepare layer inputs
        adapter_name: Name of adapter to initialize
        gather_distributed_inputs: Whether to gather across ranks
        show_progress_bar: Whether to show progress

    Returns:
        model: The model with initialized LoRA weights
    """
```

### Hook Classes

**SVDHook**: Performs incremental SVD on layer inputs

```python
class SVDHook(_Hook):
    """
    Forward hook for calculating incremental SVD on layer inputs.

    Performs incremental Singular Value Decomposition on inputs during
    forward passes and tracks convergence using cosine similarity.
    """

    def __init__(
        self,
        n_components: int,
        sim_thresh: Union[float, torch.Tensor],
        **base_class_kwargs,
    ):
        super().__init__(**base_class_kwargs)
        self.n_components = n_components
        self.sim_thresh = sim_thresh  # Convergence threshold

        # Validate sim_thresh shape
        if isinstance(sim_thresh, torch.Tensor) and len(sim_thresh.shape) > 0:
            check1 = sim_thresh.size(0) == n_components or sim_thresh.size(0) == 1
            check2 = len(sim_thresh.shape) == 1
            if not (check1 and check2):
                raise ValueError(
                    "sim_thresh must have shape (n_components,) or (1,)"
                )

        # Initialize incremental PCA
        self.svd = IncrementalPCA(
            n_components=n_components,
            copy=True,
            lowrank=True,
            lowrank_seed=42,
        )
        self.converged = torch.zeros((n_components,), dtype=torch.bool)

    @torch.no_grad()
    def __call__(self, model, input, output):
        """Hook called during forward pass"""
        previous_components = None
        if hasattr(self.svd, "components_"):
            previous_components = self.svd.components_.clone().detach()

        # Prepare and gather inputs
        states = self.prepare_layer_inputs(input)
        states = self.gather_layer_inputs(states)

        # Skip if batch size < n_components
        if states.size(0) < self.n_components:
            print(f"skipping SVD for {self.name} because batch too small")
            return

        # Perform incremental SVD step
        self.svd.partial_fit(states.to(torch.float32))

        # Check convergence
        if previous_components is None:
            return

        components = self.svd.components_
        if len(components.shape) == 1:
            components = components.reshape(1, -1)
            previous_components = previous_components.reshape(1, -1)

        # Compute cosine similarity for convergence check
        sim = torch.nn.functional.cosine_similarity(components, previous_components)
        self.converged = sim >= self.sim_thresh
```

**HashHook**: Detects layers with identical inputs

```python
class HashHook(_Hook):
    """
    Forward hook for hashing layer inputs.

    Used to detect layers receiving identical inputs, allowing
    SVD to be computed only once for such groups.
    """

    def __init__(self, **base_class_kwargs):
        super().__init__(**base_class_kwargs)
        self.hashed_inputs = []

    @staticmethod
    def hash_fn(tensor):
        """Hash tensor by converting to tuple"""
        return hash(tuple(tensor.view(-1).tolist()))

    @torch.no_grad()
    def __call__(self, model, input, output):
        """Hash inputs during forward pass"""
        x = self.prepare_layer_inputs(input)
        x = self.gather_layer_inputs(x)
        self.hashed_inputs.append(self.hash_fn(x.cpu()))
```

### Core Algorithm: _get_eva_state_dict

```python
def _get_eva_state_dict(
    model: torch.nn.Module,
    dataloader: Iterable,
    peft_config: Optional[LoraConfig],
    target_module_check_fn: callable,
    forward_fn: Optional[callable],
    prepare_model_inputs_fn: Optional[callable],
    prepare_layer_inputs_fn: Union[callable, dict[str, callable], None],
    gather_distributed_inputs: bool,
    show_progress_bar: bool,
) -> dict:
    """Internal function implementing EVA algorithm"""

    # 1. Compute rank distribution based on explained variance
    def _get_rank_distribution(hooks, layer_hook_map, equal_inputs_map, rank_budget, max_components):
        """Allocate rank budget based on explained variance ratios"""
        exp_vars = {k: h[0].svd.explained_variance_ratio_[: max_components[k]] for k, h in hooks.items()}

        # Flatten all explained variances with their layer keys
        keys, values = zip(*[(k, c) for k, name in layer_hook_map.items() for c in exp_vars[name]])

        # Sort by explained variance (descending)
        idx = torch.stack(values).argsort(descending=True)

        # Allocate rank based on top components
        counts = Counter([keys[i] for i in idx[:rank_budget]])
        counts = {k: counts.get(k, 0) for k in layer_hook_map.keys()}

        # Ensure equal input layers get consistent ranks
        for k, k_hook in equal_inputs_map.items():
            rank, rank_hook = counts[k], counts[k_hook]
            if rank_hook >= rank:
                continue
            counts[k_hook], counts[k] = rank, rank_hook

        return counts

    # 2. Validate inputs
    if len(dataloader) == 0:
        raise ValueError("dataloader is empty")

    if dist.is_initialized() and gather_distributed_inputs:
        warnings.warn(
            "torch.distributed is initialized and gather_distributed_inputs=True, "
            "EVA will gather tensors from all ranks."
        )

    # 3. Setup
    rho_threshold = 1000
    rho = peft_config.eva_config.rho
    if rho > rho_threshold:
        max_dim = max(max(p.shape) for p in model.parameters())
        rho_ceil = max_dim // peft_config.r
        rho = min(rho, rho_ceil)

    training = model.training
    device = get_device_with_meta_params(model)
    model.eval()

    # 4. Get model inputs for hooks
    inputs = next(iter(dataloader))
    if device is not None:
        inputs = move_inputs_to_device(inputs, device)
    if prepare_model_inputs_fn is not None:
        model_inputs_for_hooks = prepare_model_inputs_fn(inputs, peft_config)
    else:
        model_inputs_for_hooks = deepcopy(inputs)

    # 5. Register HashHooks to detect equal inputs
    hooks = {}
    max_components = {}
    rank_budget = 0
    for name, module in model.named_modules():
        if not target_module_check_fn(name, module):
            continue

        fn = prepare_layer_inputs_fn.pop(name, None) if isinstance(prepare_layer_inputs_fn, Mapping) else prepare_layer_inputs_fn

        hook = HashHook(name=name, prepare_layer_inputs_fn=fn, gather_distributed_inputs=gather_distributed_inputs)
        hook.model_input = model_inputs_for_hooks
        handle = module.register_forward_hook(hook)
        hooks[name] = (hook, handle)

        # Calculate max components per layer
        layer_rank = peft_config.rank_pattern.get(
            get_pattern_key(peft_config.rank_pattern.keys(), name), peft_config.r
        )
        max_components[name] = round(layer_rank * rho)
        rank_budget += layer_rank

    # 6. Forward pass to compute hashes
    forward_fn(model, inputs)
    hash_dict = {k: h[0].hashed_inputs[0] for k, h in hooks.items()}

    # 7. Find layers with equal inputs
    equal_inputs = list(find_equal_values(hash_dict).values())
    equal_inputs_map = {vv: v[0] for v in equal_inputs for vv in v[1:]}

    # Ensure equal-input layers have same max_components
    for names in equal_inputs:
        max_value = max(max_components[n] for n in names)
        for n in names:
            max_components[n] = max_value

    # 8. Replace HashHooks with SVDHooks (skip equal-input duplicates)
    for name in list(hooks.keys()):
        hook, handle = hooks.pop(name)
        handle.remove()

        if name in equal_inputs_map:
            continue  # Skip - SVD computed by representative layer

        hook = SVDHook(
            n_components=max_components[name],
            sim_thresh=peft_config.eva_config.tau,
            name=name,
            prepare_layer_inputs_fn=hook._prepare_layer_inputs_fn,
            gather_distributed_inputs=gather_distributed_inputs,
        )
        module = model.get_submodule(name)
        handle = module.register_forward_hook(hook)
        hooks[name] = (hook, handle)

    layer_hook_map = {**dict(zip(hooks.keys(), hooks.keys())), **equal_inputs_map}

    # 9. Iteratively compute SVD with convergence monitoring
    if show_progress_bar and (not dist.is_initialized() or dist.get_rank() == 0):
        pbar = tqdm(iter(cycle(dataloader)), position=0, leave=False)
        use_tqdm = True
    else:
        pbar = iter(cycle(dataloader))
        use_tqdm = False

    convergence_dict = {k: False for k in hooks.keys()}
    rank_dist = max_components.copy()

    for inputs in pbar:
        if device is not None:
            inputs = move_inputs_to_device(inputs, device)

        if prepare_model_inputs_fn is not None:
            model_inputs_for_hooks = prepare_model_inputs_fn(inputs, peft_config)
        else:
            model_inputs_for_hooks = deepcopy(inputs)

        # Update hooks with new model inputs
        for name in list(hooks.keys()):
            hook, handle = hooks[name]

            # Check convergence
            converged = torch.all(hook.converged[: rank_dist[name]])

            # Remove/re-add hook based on convergence state
            if (not convergence_dict[name]) and converged and handle:
                handle.remove()
                handle = None
                convergence_dict[name] = True
                continue
            elif convergence_dict[name] and not converged:
                module = model.get_submodule(name)
                handle = module.register_forward_hook(hook)
                convergence_dict[name] = False

            hook.model_input = model_inputs_for_hooks
            hooks[name] = (hook, handle)

        # Update progress bar
        if use_tqdm:
            layer_converged = list(convergence_dict.values()) + [
                convergence_dict[v] for v in equal_inputs_map.values()
            ]
            pbar.set_description(f"{sum(layer_converged)}/{len(layer_converged)} layers converged")

        # Stop if all converged
        if all(convergence_dict.values()):
            break

        # Forward pass for this batch
        forward_fn(model, inputs)

        # Skip if any SVD hasn't computed components yet
        if not all(hasattr(h[0].svd, "components_") for h in hooks.values()):
            continue

        # Recompute rank distribution
        rank_dist = _get_rank_distribution(hooks, layer_hook_map, equal_inputs_map, rank_budget, max_components)

    # 10. Verify all hooks removed
    remaining_hooks = {n for n, m in model.named_modules() for v in m._forward_hooks.values() if isinstance(v, _Hook)}
    if len(remaining_hooks) > 0:
        raise ValueError(f"Found active hooks that weren't removed: {remaining_hooks}")

    # 11. Build final EVA state dict
    eva_state_dict = {}
    for name, rank in rank_dist.items():
        hook = hooks[layer_hook_map[name]][0]

        if not torch.all(hook.converged[:rank]):
            raise ValueError(f"Layer {name} hasn't converged but was assigned rank {rank}")

        u = hook.svd.components_[:rank]

        # Optional whitening transformation
        if peft_config.eva_config.whiten:
            u /= hook.svd.singular_values_[:rank].sqrt().reshape(-1, 1)

        eva_state_dict[name] = u

    # 12. Restore model state
    model.train(training)

    # 13. Move tensors to device
    if device is not None:
        eva_state_dict = {k: v.to(device) for k, v in eva_state_dict.items()}

    return eva_state_dict
```

### Helper Functions

**Prepare functions for language modeling:**

```python
def prepare_model_inputs_fn_language_modeling(model_input, peft_config: LoraConfig):
    """Get indices of items to use for SVD (excluding padding)"""
    if not isinstance(model_input, dict):
        raise ValueError("inputs must be a dictionary")

    mask = model_input.get("attention_mask", torch.ones_like(model_input["input_ids"])).bool()

    if peft_config.eva_config.use_label_mask and hasattr(model_input, "labels"):
        mask = torch.logical_and(mask, model_input["labels"] != peft_config.eva_config.label_mask_value)

    return mask.nonzero()


def prepare_layer_inputs_fn_language_modeling(layer_input, model_input, layer_name) -> torch.Tensor:
    """Extract layer inputs using mask from prepare_model_inputs_fn"""
    if isinstance(layer_input, torch.Tensor):
        pass
    elif isinstance(layer_input, (tuple, list)):
        layer_input = layer_input[0]
    else:
        raise ValueError(f"unsupported input type {type(layer_input)}")

    # model_input is the output of prepare_model_inputs_fn_language_modeling
    return layer_input[model_input.T.unbind()]
```

## I/O Contract

### get_eva_state_dict()

**Inputs:**
- `model` (torch.nn.Module): Model to compute SVD for
- `dataloader` (Iterable): Dataloader providing calibration data
- `peft_config` (Optional[LoraConfig]): LoRA configuration (required if not PeftModel)
- `forward_fn` (callable): Function signature `forward_fn(model, inputs)`
- `prepare_model_inputs_fn` (Optional[callable]): Preprocesses model inputs for hooks
- `prepare_layer_inputs_fn` (callable | dict): Preprocesses layer inputs for SVD
- `adapter_name` (str): Adapter name to compute SVD for
- `gather_distributed_inputs` (bool): Whether to gather across distributed ranks
- `show_progress_bar` (bool): Display progress bar

**Outputs:**
- `dict`: Maps layer names to SVD components (torch.Tensor), shape `(rank, in_features)`

**Side Effects:**
- Temporarily sets model to eval mode
- Registers/removes forward hooks
- May gather tensors across distributed ranks

**Convergence Requirements:**
- Requires sufficient distinct samples (typically 256+ for hidden_dim=4096, seq_len=2048)
- Rule of thumb: `samples ≈ hidden_dim / seq_len * 128`
- Will iterate until all layers converge or dataloader exhausted

### initialize_lora_eva_weights()

**Inputs:**
- `model` (torch.nn.Module): PeftModel with `init_lora_weights='eva'`
- `dataloader` (Optional[Iterable]): Calibration data (required if eva_state_dict=None)
- `eva_state_dict` (Optional[dict]): Pre-computed state dict
- `forward_fn`, `prepare_model_inputs_fn`, `prepare_layer_inputs_fn`: Same as get_eva_state_dict
- `adapter_name` (str): Adapter to initialize
- `gather_distributed_inputs` (bool): Gather across ranks
- `show_progress_bar` (bool): Display progress

**Outputs:**
- Returns the same `model` object (modified in-place)

**Side Effects:**
- Initializes lora_A weights with EVA components
- Updates model.peft_config rank_pattern and alpha_pattern
- May remove adapters with rank=0
- Updates target_modules list

**Constraints:**
- Only works with single active adapter
- Requires `init_lora_weights='eva'` in config
- Model must be a PeftModel

## Usage Examples

### Basic EVA Initialization

```python
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.eva import initialize_lora_eva_weights
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure LoRA with EVA initialization
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    init_lora_weights="eva",  # Enable EVA
    eva_config={
        "rho": 2.0,         # Max rank multiplier for SVD computation
        "tau": 0.95,        # Convergence threshold (cosine similarity)
        "whiten": True,     # Apply whitening transformation
    }
)

# Create PEFT model
peft_model = get_peft_model(model, config)

# Prepare calibration dataloader
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

# Assume 'dataset' is your calibration dataset
dataloader = DataLoader(dataset.map(tokenize_function), batch_size=4)

# Initialize with EVA
initialize_lora_eva_weights(
    peft_model,
    dataloader=dataloader,
    adapter_name="default",
    show_progress_bar=True
)

# Model is now ready for fine-tuning with EVA-initialized weights
```

### Computing and Reusing EVA State Dict

```python
from peft.tuners.lora.eva import get_eva_state_dict, initialize_lora_eva_weights
import torch

# Compute EVA state dict once
eva_state_dict = get_eva_state_dict(
    model=peft_model,
    dataloader=dataloader,
    show_progress_bar=True,
    gather_distributed_inputs=True
)

# Save for later use
torch.save(eva_state_dict, "eva_state_dict.pt")

# Later: load and apply to new model
eva_state_dict = torch.load("eva_state_dict.pt")
new_model = get_peft_model(base_model, config)

initialize_lora_eva_weights(
    new_model,
    eva_state_dict=eva_state_dict,  # Use pre-computed state dict
    adapter_name="default"
)
```

### Custom Preprocessing for Non-Language Models

```python
def custom_prepare_model_inputs_fn(model_input, peft_config):
    """Custom preprocessing for vision models"""
    # Return all inputs (no masking)
    batch_size, channels, height, width = model_input.shape
    return torch.arange(batch_size * height * width).reshape(batch_size, -1)

def custom_prepare_layer_inputs_fn(layer_input, model_input, layer_name):
    """Custom layer input preparation"""
    if isinstance(layer_input, (tuple, list)):
        layer_input = layer_input[0]

    # Reshape if needed
    if layer_input.ndim > 2:
        layer_input = layer_input.reshape(-1, layer_input.shape[-1])

    return layer_input

# Use custom functions
eva_state_dict = get_eva_state_dict(
    model=vision_model,
    dataloader=vision_dataloader,
    prepare_model_inputs_fn=custom_prepare_model_inputs_fn,
    prepare_layer_inputs_fn=custom_prepare_layer_inputs_fn,
)
```

### Per-Layer Custom Preprocessing

```python
# Define different preprocessing for different layers
prepare_layer_inputs_dict = {
    "model.layers.0.self_attn.q_proj": lambda inp, model_inp, name: custom_fn_1(inp),
    "model.layers.0.self_attn.v_proj": lambda inp, model_inp, name: custom_fn_2(inp),
    # Other layers use default
}

eva_state_dict = get_eva_state_dict(
    model=peft_model,
    dataloader=dataloader,
    prepare_layer_inputs_fn=prepare_layer_inputs_dict,  # Dict for per-layer customization
)
```

### Distributed Training Setup

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize distributed
dist.init_process_group(backend="nccl")
rank = dist.get_rank()

# Setup model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = model.to(rank)
model = DDP(model, device_ids=[rank])

# Create PEFT model
peft_model = get_peft_model(model.module, config)

# EVA with distributed input gathering
eva_state_dict = get_eva_state_dict(
    model=peft_model,
    dataloader=distributed_dataloader,  # Each rank sees different data
    gather_distributed_inputs=True,     # Gather across ranks for better statistics
    show_progress_bar=(rank == 0),      # Only show on rank 0
)
```

### Monitoring Convergence

```python
from peft.tuners.lora.eva import SVDHook
import matplotlib.pyplot as plt

# Track convergence manually
convergence_history = []

class MonitoredSVDHook(SVDHook):
    def __call__(self, model, input, output):
        super().__call__(model, input, output)
        convergence_history.append({
            'layer': self.name,
            'converged': self.converged.clone(),
            'step': len(convergence_history)
        })

# Use custom hook (requires modifying get_eva_state_dict internally)
# Or analyze convergence from progress bar updates

# After EVA computation
for entry in convergence_history:
    print(f"Step {entry['step']}: {entry['layer']} - {entry['converged'].sum()}/{len(entry['converged'])} converged")
```

### Adjusting Rank Budget and Scaling

```python
config = LoraConfig(
    r=16,                    # Base rank
    lora_alpha=32,           # Base alpha
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    init_lora_weights="eva",
    eva_config={
        "rho": 3.0,          # Compute up to 3*r components per layer
        "tau": 0.98,         # Stricter convergence (higher = stricter)
        "whiten": True,
        "adjust_scaling_factors": True,  # Adjust alpha proportionally to rank changes
    }
)

peft_model = get_peft_model(model, config)
initialize_lora_eva_weights(peft_model, dataloader)

# Check resulting rank distribution
for name, module in peft_model.named_modules():
    if hasattr(module, 'lora_A'):
        for adapter_name in module.lora_A.keys():
            rank = module.lora_A[adapter_name].weight.shape[0]
            print(f"{name}: rank={rank}")
```

## Related Pages

### Core LoRA Components
- `huggingface_peft_LoraConfig.md` - Configuration including eva_config
- `huggingface_peft_LoraModel.md` - Base LoRA model implementation
- `huggingface_peft_LoraLayer.md` - LoRA layer base class

### Initialization Methods
- `huggingface_peft_pissa_init.md` - PiSSA initialization (SVD of pretrained weights)
- `huggingface_peft_olora_init.md` - OLoRA initialization
- `huggingface_peft_loftq_init.md` - LoftQ initialization for quantized models
- `huggingface_peft_corda_init.md` - CorDA initialization with covariance

### Related Utilities
- `huggingface_peft_IncrementalPCA.md` - Incremental PCA implementation used by EVA
- `huggingface_peft_get_pattern_key.md` - Pattern matching for rank_pattern
- `huggingface_peft_dequantize_module_weight.md` - Weight dequantization utilities

### Configuration
- `huggingface_peft_EVAConfig.md` - EVA-specific configuration dataclass

### Concepts
- Singular Value Decomposition (SVD) for dimensionality reduction
- Incremental PCA for streaming SVD computation
- Explained variance ratio for rank selection
- Data-driven initialization vs random initialization
- Convergence detection via cosine similarity
