---
library: huggingface_peft
module: src/peft/tuners/lora/corda.py
classes: ["preprocess_corda", "CordaEigens"]
type: implementation
tags: ["lora", "corda", "covariance", "svd", "initialization", "correlation-aware"]
description: CorDA (Correlation-aware Data-driven) initialization using covariance-adjusted SVD for LoRA
version: c6db49996c63
language: en
---

# CorDA: Correlation-aware Data-driven Initialization

## Overview

CorDA (Correlation-aware Data-driven initialization) is an advanced LoRA initialization method that computes SVD on pretrained weights adjusted by input covariance matrices. Unlike standard SVD initialization (PiSSA) or activation-based methods (EVA), CorDA explicitly accounts for input correlations to produce more effective low-rank decompositions.

Key features:
- **Covariance-Adjusted SVD**: Computes SVD on W @ C where C is the input covariance matrix
- **Improved Principal Mode (IPM)**: Uses top-k singular vectors for initialization
- **Knockout Principal Mode (KPM)**: Uses bottom-k singular vectors (orthogonal to pretrained weight's principal modes)
- **Memory Efficient**: Requires ~2x model memory temporarily for covariance computation
- **Cache Support**: Can save/load covariance matrices and eigenvectors for reuse

The implementation builds three key components per layer:
- **Covariance Matrix**: Input activation statistics collected during calibration
- **SVD Components**: U, S, V from covariance-adjusted weight matrix
- **Cropped Eigenvectors**: Final initialization matrices stored in `module.eigens`

**Source File:** `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/lora/corda.py` (360 lines)
**Reference Paper:** https://huggingface.co/papers/2406.05223
**Reference Code:** https://github.com/iboing/CorDA/blob/main/cordalib/decomposition.py

## Code Reference

### Main API Function

**preprocess_corda**: Complete CorDA preprocessing pipeline

```python
@torch.no_grad()
def preprocess_corda(
    model: nn.Module,
    lora_config: LoraConfig,
    run_model: Optional[Callable[[], None]] = None,
    hooked_model: Optional[nn.Module] = None,
):
    """
    Build necessary CorDA fields for a model.

    For each M*N linear layer, builds an M*M covariance matrix temporarily,
    consuming roughly 2*MODEL_SIZE memory for typical LLMs (FP16 weights, FP32 covariance).

    Args:
        model (nn.Module): Model to preprocess
        lora_config (LoraConfig): Configuration with corda_config set
        run_model (Optional[Callable]): Callback to run inference on calibration data.
            Should run ~256 distinct samples (2048 tokens each for hidden_dim=4096).
            Sample count estimate: HIDDEN_DIM / TOKEN_PER_SAMPLE * 128.
            Required if cache_file doesn't exist.
        hooked_model (Optional[nn.Module]): Alternative model to hook (rarely needed)

    Upon completion, sets for each target module:
        eigens.S_WC (torch.Tensor): Singular values
        eigens.U_WC (torch.Tensor): Left singular vectors
        eigens.V_WC (torch.Tensor): Right singular vectors * inv(covariance)
    """
    cache_file = lora_config.corda_config.cache_file
    covariance_file = lora_config.corda_config.covariance_file
    corda_method = lora_config.corda_config.corda_method
    verbose = lora_config.corda_config.verbose
    prune_temporary_fields = lora_config.corda_config.prune_temporary_fields

    # Load from cache if exists
    if cache_file is not None and os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        cache = torch.load(cache_file, map_location=get_model_device(model))
        for name, module in target_modules(model, lora_config):
            module.eigens = CordaEigens(
                S_WC=cache[f"{name}.eigens.S_WC"],
                U_WC=cache[f"{name}.eigens.U_WC"],
                V_WC=cache[f"{name}.eigens.V_WC"],
            )
    else:
        # Full preprocessing pipeline
        if corda_method is None:
            raise ValueError("corda_method is required when cache_file is not provided.")

        # Set CorDA method for each layer
        for name, module in target_modules(model, lora_config):
            module.corda_method = corda_method

        # Set rank for each layer
        for name, module in target_modules(model, lora_config):
            r_key = get_pattern_key(lora_config.rank_pattern.keys(), name)
            module.rank = lora_config.rank_pattern.get(r_key, lora_config.r)

        # Calculate covariance matrix
        calib_cov_distribution(model, lora_config, run_model, hooked_model, covariance_file)

        # Calculate eigens (SVD)
        collect_eigens(model, lora_config, verbose)

        # Crop eigens to configured rank
        crop_corda_eigens(model, lora_config)

        # Remove redundant fields
        if prune_temporary_fields:
            for name, module in target_modules(model, lora_config):
                if hasattr(module, "sample_count"):
                    del module.sample_count
                if hasattr(module, "covariance_matrix"):
                    del module.covariance_matrix
                if hasattr(module, "corda_method"):
                    del module.corda_method
                if hasattr(module, "rank"):
                    del module.rank

        # Save cache
        if cache_file is not None:
            cache: dict[str, Any] = {}
            for name, module in target_modules(model, lora_config):
                cache[f"{name}.eigens.S_WC"] = module.eigens.S_WC
                cache[f"{name}.eigens.U_WC"] = module.eigens.U_WC
                cache[f"{name}.eigens.V_WC"] = module.eigens.V_WC

            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            torch.save(cache, cache_file)
```

### Data Structures

**CordaEigens**: Stores SVD components for a layer

```python
@dataclass
class CordaEigens:
    S_WC: torch.Tensor  # Singular values, shape (rank,)
    U_WC: torch.Tensor  # Left singular vectors, shape (out_features, rank)
    V_WC: torch.Tensor  # Right singular vectors * inv(C), shape (in_features, rank)
```

### Covariance Computation

**calib_cov_distribution**: Collect input covariance matrices

```python
@torch.no_grad()
def calib_cov_distribution(
    model: nn.Module,
    config: LoraConfig,
    run_model: Optional[Callable[[], None]],
    hooked_model: Optional[nn.Module],
    covariance_file: Optional[str],
):
    """Calculate or load covariance matrices for all target modules"""

    # Load from file if exists
    if covariance_file is not None and os.path.exists(covariance_file) and os.path.getsize(covariance_file) > 0:
        all_covariance_matrix = torch.load(covariance_file, map_location=get_model_device(model))
        for name, module in target_modules(model, config):
            module.covariance_matrix = all_covariance_matrix[name]
        return

    # Validate inputs
    if run_model is None:
        raise ValueError("run_model must be specified when covariance file doesn't exist")
    if hooked_model is None:
        hooked_model = model
    hooked_model.eval()

    # Define hook to accumulate covariance
    def hook(module, input, output):
        """Forward hook that computes X^T @ X where X is layer input"""
        input = input[0].detach().squeeze(0).data  # (context_length, dim)

        # Cast to float32 unless explicitly using float16
        if not config.corda_config.use_float16_for_covariance:
            input = input.float()

        # Normalize by max absolute value
        input = input / torch.max(input).abs()

        # Validate input
        if torch.isnan(input).any() or torch.isinf(input).any():
            raise ValueError("Invalid value in input, check your data")

        # Compute covariance
        covariance = input.t().matmul(input)
        if torch.isnan(covariance).any() or torch.isinf(covariance).any():
            raise ValueError("Invalid value in covariance, please file an issue")

        # Accumulate
        module.sample_count += 1
        module.covariance_matrix += covariance

        del covariance, input

    # Register hooks
    handles = []
    for name, module in target_modules(hooked_model, config):
        module.sample_count = 0
        module.covariance_matrix = 0
        handles.append(module.register_forward_hook(hook))

    # Run calibration
    run_model()

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Transfer from hooked_model to model if different
    if hooked_model is not model:
        targets = {}
        for name, module in target_modules(model, config):
            targets[name] = module
        for name, module in target_modules(hooked_model, config):
            if name in targets:
                targets[name].sample_count = module.sample_count
                targets[name].covariance_matrix = module.covariance_matrix

    # Average over samples
    for name, module in target_modules(model, config):
        module.covariance_matrix /= module.sample_count

    # Save to file
    if covariance_file is not None:
        all_covariance_matrix = {}
        for name, module in target_modules(model, config):
            all_covariance_matrix[name] = module.covariance_matrix
        os.makedirs(os.path.dirname(covariance_file), exist_ok=True)
        torch.save(all_covariance_matrix, covariance_file)
```

### SVD Computation

**collect_eigens_for_layer**: Compute covariance-adjusted SVD for a single layer

```python
@torch.no_grad()
def collect_eigens_for_layer(
    linear: nn.Linear,
    config: LoraConfig,
) -> CordaEigens:
    """Compute SVD of W @ C where C is covariance matrix"""

    w = linear.weight.data.float()  # (out_dim, in_dim)
    out_dim = w.size(0)
    in_dim = w.size(1)
    min_dim = min(in_dim, out_dim)

    if not hasattr(linear, "covariance_matrix"):
        raise ValueError("Covariance matrix not found, call preprocess_corda")

    covariance_matrix = linear.covariance_matrix.float()

    # Stabilize covariance inversion with damping
    damp = 0.01
    while True:
        # Add diagonal compensation
        compensate = torch.diag(
            torch.ones(covariance_matrix.size(0)).to(covariance_matrix.device)
            * torch.mean(torch.diag(covariance_matrix))
            * damp
        )
        fix_covariance_matrix = covariance_matrix + compensate

        # Compute inverse
        cov_inv = torch.linalg.inv(fix_covariance_matrix)

        # Check inversion error
        inv_error = torch.dist(
            fix_covariance_matrix @ cov_inv,
            torch.eye(covariance_matrix.size(0)).to(get_model_device(linear))
        ).item()

        if inv_error < 0.05:
            break  # Acceptable error
        else:
            damp = damp * 2  # Increase damping

    # Compute W @ C
    w = w @ fix_covariance_matrix  # (out_dim, in_dim)

    # SVD: W @ C = U @ S @ Vh
    U, S, Vh = torch.linalg.svd(w, full_matrices=False)

    # V = Vh^T @ C^{-1}
    V = (Vh @ cov_inv).transpose(0, 1)  # (in_dim, min_dim)

    # Sanity checks
    r = min_dim
    if U.size(0) != out_dim or U.size(1) != r:
        raise ValueError(f"Matrix U size mismatch: {U.size()} vs. ({out_dim}, {r})")
    if S.size(0) != r:
        raise ValueError(f"Matrix S size mismatch: {S.size()} vs. ({r},)")
    if V.size(0) != in_dim or V.size(1) != r:
        raise ValueError(f"Matrix V size mismatch: {V.size()} vs. ({in_dim}, {r})")

    # Offload to CPU to save memory
    U = U.cpu()
    V = V.cpu()

    return CordaEigens(S_WC=S, U_WC=U, V_WC=V)
```

**collect_eigens**: Compute eigens for all layers

```python
@torch.no_grad()
def collect_eigens(
    model: nn.Module,
    config: LoraConfig,
    verbose: bool,
):
    """Call collect_eigens_for_layer for each target module"""
    linear_modules = []
    for name, module in target_modules(model, config):
        linear_modules.append((name, module))

    if verbose:
        linear_modules = tqdm(linear_modules, desc="Collecting eigens")

    for name, module in linear_modules:
        module.eigens = collect_eigens_for_layer(module, config)
```

### Eigen Cropping

**crop_corda_eigens**: Crop eigens to configured rank using IPM or KPM

```python
@torch.no_grad()
def crop_corda_eigens(model: nn.Module, config: LoraConfig):
    """Crop eigens to specified rank based on corda_method"""
    for name, module in target_modules(model, config):
        # Clone tensors to avoid saving full matrices
        # Reference: https://github.com/pytorch/pytorch/issues/40157
        if module.corda_method == "ipm":
            # Improved Principal Mode: top-k singular vectors
            module.eigens.S_WC = module.eigens.S_WC[: module.rank].clone()
            module.eigens.U_WC = module.eigens.U_WC[:, : module.rank].clone().to(get_model_device(model))
            module.eigens.V_WC = module.eigens.V_WC[:, : module.rank].clone().to(get_model_device(model))
        elif module.corda_method == "kpm":
            # Knockout Principal Mode: bottom-k singular vectors
            module.eigens.S_WC = module.eigens.S_WC[-module.rank :].clone()
            module.eigens.U_WC = module.eigens.U_WC[:, -module.rank :].clone().to(get_model_device(model))
            module.eigens.V_WC = module.eigens.V_WC[:, -module.rank :].clone().to(get_model_device(model))
        else:
            raise ValueError(f"Invalid corda_method: {module.corda_method}, must be 'ipm' or 'kpm'")

        # Sanity checks
        if module.eigens.S_WC.size(0) != module.rank:
            raise ValueError(f"rank mismatch: {module.eigens.S_WC.size(0)} vs. {module.rank}")
        if module.eigens.U_WC.size(0) != module.weight.size(0):
            raise ValueError(f"U size mismatch: {module.eigens.U_WC.size(0)} vs. {module.weight.size(0)}")
        if module.eigens.U_WC.size(1) != module.rank:
            raise ValueError(f"U size mismatch: {module.eigens.U_WC.size(1)} vs. {module.rank}")
        if module.eigens.V_WC.size(0) != module.weight.size(1):
            raise ValueError(f"V size mismatch: {module.eigens.V_WC.size(0)} vs. {module.weight.size(1)}")
        if module.eigens.V_WC.size(1) != module.rank:
            raise ValueError(f"V size mismatch: {module.eigens.V_WC.size(1)} vs. {module.rank}")
```

### Helper Functions

**target_modules**: Iterate over CorDA-eligible modules

```python
def target_modules(model: nn.Module, config: LoraConfig) -> Iterable[nn.Module]:
    """
    Iterate over CorDA target modules. A module is targeted if:
    - Its name is in config.target_modules
    - It is an nn.Linear layer
    """
    for name, module in model.named_modules():
        if LoraModel._check_target_module_exists(config, name) and isinstance(module, nn.Linear):
            yield name, module


def get_model_device(model: nn.Module) -> str:
    """Get device of model parameters (handles DeepSpeed/DataParallel)"""
    if hasattr(model, "module"):
        model = model.module
    return next(iter(model.parameters())).device.type
```

## I/O Contract

### preprocess_corda()

**Inputs:**
- `model` (nn.Module): Model to preprocess (modified in-place)
- `lora_config` (LoraConfig): Configuration with `corda_config` set
- `run_model` (Optional[Callable[[], None]]): Callback to run calibration inference
  - Should iterate over ~256 distinct samples
  - For hidden_dim=4096, token_per_sample=2048: 256 samples recommended
  - Estimate: `samples ≈ hidden_dim / token_per_sample * 128`
- `hooked_model` (Optional[nn.Module]): Alternative model to hook (rarely used)

**Outputs:**
- None (modifies model in-place)

**Side Effects:**
- Adds `module.eigens` (CordaEigens) to each target module
- Creates covariance_file if specified and doesn't exist
- Creates cache_file if specified
- Temporarily uses ~2x model memory for covariance matrices
- Removes temporary fields if `prune_temporary_fields=True`

**Memory Requirements:**
- Peak: ~3x model size (model + covariance + SVD intermediates)
- After pruning: ~1.1x model size (model + small eigens)

### calib_cov_distribution()

**Inputs:**
- `model` (nn.Module): Model to compute covariance for
- `config` (LoraConfig): Configuration
- `run_model` (Optional[Callable]): Calibration callback
- `hooked_model` (Optional[nn.Module]): Model to hook
- `covariance_file` (Optional[str]): Path to save/load covariance

**Outputs:**
- None (modifies modules in-place)

**Side Effects:**
- Adds `module.covariance_matrix` and `module.sample_count` to target modules
- Saves covariance_file if specified
- Registers/removes forward hooks

### collect_eigens_for_layer()

**Inputs:**
- `linear` (nn.Linear): Linear layer with `covariance_matrix` attribute
- `config` (LoraConfig): Configuration

**Outputs:**
- `CordaEigens`: SVD components (U, S, V) in full rank

**Computational Complexity:**
- Covariance inversion: O(d³) where d=in_features
- SVD: O(min(m,n)²·max(m,n)) where m=out_features, n=in_features
- Dominant for typical LLMs: d=4096, complexity ~68B FLOPs per layer

### crop_corda_eigens()

**Inputs:**
- `model` (nn.Module): Model with full-rank eigens
- `config` (LoraConfig): Configuration with rank and corda_method

**Outputs:**
- None (modifies `module.eigens` in-place)

**Side Effects:**
- Crops eigens to configured rank
- Moves eigens back to model device (from CPU)
- Validates all dimensions

## Usage Examples

### Basic CorDA Initialization (IPM)

```python
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.corda import preprocess_corda
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure LoRA with CorDA
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    init_lora_weights="corda",  # Enable CorDA initialization
    corda_config={
        "corda_method": "ipm",              # Improved Principal Mode (top-k)
        "cache_file": "./corda_cache.pt",   # Cache for reuse
        "covariance_file": "./cov.pt",      # Covariance cache
        "verbose": True,                     # Show progress
        "prune_temporary_fields": True,      # Clean up after
        "use_float16_for_covariance": False, # Use FP32 for stability
    }
)

# Create PEFT model
peft_model = get_peft_model(model, config)

# Define calibration function
def run_calibration():
    for batch in calibration_dataloader:
        inputs = tokenizer(batch["text"], return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            peft_model(**inputs)

# Preprocess with CorDA
preprocess_corda(
    model=peft_model.base_model.model,  # Access base model
    lora_config=config,
    run_model=run_calibration
)

# Model is now ready with CorDA-initialized weights
```

### KPM (Knockout Principal Mode)

```python
# Use bottom-k singular vectors instead of top-k
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    init_lora_weights="corda",
    corda_config={
        "corda_method": "kpm",  # Knockout Principal Mode (bottom-k)
        "cache_file": "./corda_kpm_cache.pt",
    }
)

peft_model = get_peft_model(model, config)
preprocess_corda(peft_model.base_model.model, config, run_calibration)
```

### Reusing Cached CorDA State

```python
# First run: compute and cache
config1 = LoraConfig(
    r=8,
    init_lora_weights="corda",
    corda_config={
        "corda_method": "ipm",
        "cache_file": "./corda_r8.pt",
        "covariance_file": "./cov_shared.pt",
    }
)
peft_model1 = get_peft_model(model, config1)
preprocess_corda(peft_model1.base_model.model, config1, run_calibration)

# Second run: reuse covariance, compute new eigens for different rank
config2 = LoraConfig(
    r=16,  # Different rank
    init_lora_weights="corda",
    corda_config={
        "corda_method": "ipm",
        "cache_file": "./corda_r16.pt",
        "covariance_file": "./cov_shared.pt",  # Reuse covariance!
    }
)
peft_model2 = get_peft_model(model, config2)
# No need to run calibration again, covariance loaded from file
preprocess_corda(peft_model2.base_model.model, config2, run_model=None)
```

### Memory-Constrained Setup

```python
# Use FP16 for covariance to reduce memory (may be less stable)
config = LoraConfig(
    r=8,
    init_lora_weights="corda",
    corda_config={
        "corda_method": "ipm",
        "use_float16_for_covariance": True,  # Reduce memory ~50%
        "prune_temporary_fields": True,       # Clean up aggressively
    }
)

peft_model = get_peft_model(model, config)
preprocess_corda(peft_model.base_model.model, config, run_calibration)
```

### Custom Rank Per Layer

```python
config = LoraConfig(
    r=16,  # Default rank
    rank_pattern={
        "q_proj": 32,    # Higher rank for query
        "k_proj": 8,     # Lower rank for key
        "v_proj": 16,    # Default rank
        "o_proj": 16,
    },
    init_lora_weights="corda",
    corda_config={
        "corda_method": "ipm",
        "cache_file": "./corda_varied_rank.pt",
    }
)

peft_model = get_peft_model(model, config)
preprocess_corda(peft_model.base_model.model, config, run_calibration)

# Verify ranks
for name, module in peft_model.named_modules():
    if hasattr(module, 'eigens'):
        print(f"{name}: rank={module.eigens.U_WC.shape[1]}")
```

### Inspecting Covariance and Eigens

```python
import torch

# After preprocessing
for name, module in peft_model.named_modules():
    if hasattr(module, 'eigens'):
        print(f"\n{name}:")
        print(f"  Singular values: {module.eigens.S_WC}")
        print(f"  U shape: {module.eigens.U_WC.shape}")  # (out_features, rank)
        print(f"  V shape: {module.eigens.V_WC.shape}")  # (in_features, rank)

        # Analyze singular value distribution
        S = module.eigens.S_WC
        print(f"  Top 5 singular values: {S[:5].tolist()}")
        print(f"  Singular value ratio (top/bottom): {S[0]/S[-1]:.2f}")

# Load and inspect covariance directly
if os.path.exists("./cov.pt"):
    covariances = torch.load("./cov.pt")
    for name, cov in covariances.items():
        print(f"{name} covariance: shape={cov.shape}, mean={cov.mean():.4f}")
```

### Comparing IPM vs KPM

```python
# Run both methods and compare performance
for method in ["ipm", "kpm"]:
    config = LoraConfig(
        r=16,
        init_lora_weights="corda",
        corda_config={
            "corda_method": method,
            "cache_file": f"./corda_{method}.pt",
            "covariance_file": "./cov_shared.pt",
        }
    )

    model_fresh = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    peft_model = get_peft_model(model_fresh, config)

    # First iteration computes covariance, second reuses
    if method == "ipm":
        preprocess_corda(peft_model.base_model.model, config, run_calibration)
    else:
        preprocess_corda(peft_model.base_model.model, config, run_model=None)

    # Fine-tune and evaluate...
    print(f"Method: {method}")
    # ... training code ...
```

### Hooked Model Pattern (Advanced)

```python
# Use a different model for hooking (e.g., for special distributed setups)
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
inference_model = deepcopy(base_model)

config = LoraConfig(
    r=16,
    init_lora_weights="corda",
    corda_config={"corda_method": "ipm"}
)

peft_model = get_peft_model(base_model, config)

def run_inference_model():
    for batch in calibration_dataloader:
        inputs = tokenizer(batch["text"], return_tensors="pt")
        with torch.no_grad():
            inference_model(**inputs)

# Hooks attached to inference_model, covariance transferred to peft_model
preprocess_corda(
    model=peft_model.base_model.model,
    lora_config=config,
    run_model=run_inference_model,
    hooked_model=inference_model
)
```

## Related Pages

### Core LoRA Components
- `huggingface_peft_LoraConfig.md` - Configuration including corda_config
- `huggingface_peft_LoraModel.md` - Base LoRA model implementation
- `huggingface_peft_LoraLayer.md` - LoRA layer base class

### Other Initialization Methods
- `huggingface_peft_EVA.md` - Activation-aware SVD initialization
- `huggingface_peft_pissa_init.md` - Standard SVD of pretrained weights
- `huggingface_peft_olora_init.md` - Orthogonal LoRA initialization
- `huggingface_peft_loftq_init.md` - LoftQ for quantized models

### Configuration
- `huggingface_peft_CordaConfig.md` - CorDA-specific configuration dataclass

### Utilities
- `huggingface_peft_get_pattern_key.md` - Pattern matching for rank_pattern
- `huggingface_peft_transpose.md` - Weight transposition utilities

### Concepts
- Singular Value Decomposition (SVD) for low-rank approximation
- Covariance matrices in neural network analysis
- Principal component analysis (PCA)
- Improved vs Knockout principal modes
- Matrix damping for numerical stability
