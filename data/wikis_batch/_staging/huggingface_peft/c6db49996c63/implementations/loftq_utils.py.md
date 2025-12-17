# Implementation: loftq_utils.py

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/utils/loftq_utils.py`
- **Size**: 410 lines
- **Module**: `peft.utils.loftq_utils`
- **Description**: LoftQ quantization-aware initialization for LoRA adapters

## Overview

This module implements LoftQ (Low-rank adaptation via Quantization), a technique for initializing LoRA adapters that are aware of weight quantization. When using quantized models (e.g., 4-bit via bitsandbytes), standard LoRA initialization can lead to poor performance due to quantization error. LoftQ addresses this by alternating between quantization and SVD decomposition to find optimal low-rank approximations that minimize the combined quantization and approximation error.

**Reference**: [LoftQ Paper](https://huggingface.co/papers/2310.08659)

## Core Classes

### NFQuantizer

**Purpose**: Normal Float quantizer for 2/4/8-bit quantization

```python
class NFQuantizer:
    def __init__(
        self,
        num_bits=2,           # Number of bits (2, 4, or 8)
        device="cuda",        # Device for computation
        method="normal",      # "normal" or "uniform" distribution
        block_size=64,        # Block size for block-wise quantization
    )
```

#### Quantization Methods

**1. Normal Distribution Quantization** (default):
```python
@staticmethod
def create_normal_map(offset=0.9677083, symmetric=False, num_bits=2):
    """
    Creates quantization codebook based on normal distribution.

    Process:
    1. Use inverse CDF (norm.ppf) to get evenly-spaced quantiles
    2. Asymmetric: more positive values (better for weights)
    3. Normalize to [-1, 1] range
    """
```

**2. Uniform Quantization**:
```python
@staticmethod
def create_uniform_map(symmetric=False, num_bits=4):
    """
    Creates linearly-spaced quantization levels.

    Symmetric: Equal positive/negative levels
    Asymmetric: Full [0, 1] range utilization
    """
```

#### Core Operations

**Tensor-wise Quantization**:
```python
def quantize_tensor(self, weight):
    """
    Quantizes entire tensor using lookup table.

    Process:
    1. Normalize: weight_norm = weight / max(abs(weight))
    2. Find nearest quantization level
    3. Return quantized indices and scale factor
    """
```

**Block-wise Quantization**:
```python
def quantize_block(self, weight):
    """
    Quantizes tensor in blocks for better accuracy.

    Process:
    1. Reshape into blocks of size block_size
    2. Compute per-block scale factors
    3. Quantize each block independently
    4. Pack multiple k-bit values into uint8

    Packing: [01, 00, 11, 10] → [10110001] (LIFO order)
    """
```

**Dequantization**:
```python
def dequantize_block(self, qweight, weight_max, weight_shape):
    """
    Recovers float weights from quantized representation.

    Process:
    1. Unpack uint8 into individual quantization indices
    2. Lookup dequantized values from table
    3. Scale by per-block maximum
    4. Reshape to original shape
    """
```

### SafetensorLoader

**Purpose**: Efficient loading of model weights from safetensors files

```python
class _SafetensorLoader:
    def __init__(self, peft_model, model_path):
        """
        Initializes loader with automatic path resolution.

        Handles:
        - Local model paths
        - HuggingFace Hub paths
        - Sharded model files
        - Base model prefix normalization
        """
```

#### Key Features

**Automatic Sharding Detection**:
```python
if not os.path.exists(model_path):
    # Check for sharded files
    resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
        par_dir, cached_file(par_dir, "model.safetensors.index.json")
    )
    self.is_sharded = True
    self.weight_map = {k: file_map[v] for k, v in sharded_metadata["weight_map"].items()}
```

**Lazy Loading**:
```python
def get_tensor(self, name):
    """
    Loads single tensor without loading entire model.

    Benefits:
    - Memory efficient
    - Fast for sparse access
    - Supports sharded models
    """
```

## Core Algorithms

### Low-Rank Decomposition

**_low_rank_decomposition(weight, reduced_rank)**
```python
def _low_rank_decomposition(weight, reduced_rank=32):
    """
    Computes SVD and returns low-rank factors.

    Mathematical Form:
    W ≈ U @ sqrt(Σ) @ sqrt(Σ) @ V^T = L @ R

    Where:
    - L = U @ sqrt(Σ)[:, :r]  (left factor)
    - R = sqrt(Σ)[:r, :] @ V^T  (right factor)

    Returns: L, R, and full SVD components
    """
```

### LoftQ Initialization (Legacy)

**loftq_init(weight, num_bits, reduced_rank, num_iter=1)**
```python
@torch.no_grad()
def loftq_init(weight, num_bits, reduced_rank, num_iter=1):
    """
    Alternating optimization for quantization-aware initialization.

    Algorithm:
    For i in range(num_iter):
        1. Quantize residual: Q = quantize(residual)
        2. Compute error: residual = weight - dequantize(Q)
        3. SVD decomposition: L, R = low_rank_decomp(residual)
        4. Update residual: residual = weight - L @ R

    Returns: quantized_weight, lora_A, lora_B
    """
```

**Key Insight**: Alternating between quantization and low-rank approximation finds a solution where:
- Quantized weight + LoRA ≈ original weight
- Quantization error is minimized by LoRA adaptation

### LoftQ Initialization (New)

**_loftq_init_new(qweight, weight, num_bits, reduced_rank)**
```python
@torch.no_grad()
def _loftq_init_new(qweight, weight, num_bits, reduced_rank):
    """
    Single-step LoftQ initialization for already-quantized weights.

    Use Case: When model is already loaded in 4-bit

    Process:
    1. Dequantize existing quantized weights
    2. Compute residual = original - dequantized
    3. SVD on residual to get LoRA factors

    Returns: lora_A (R), lora_B (L)
    """
```

### Replace LoRA Weights with LoftQ

**replace_lora_weights_loftq(peft_model, model_path, adapter_name, callback)**
```python
@torch.no_grad()
def replace_lora_weights_loftq(
    peft_model,
    model_path: Optional[str] = None,
    adapter_name: str = "default",
    callback: Optional[Callable[[torch.nn.Module, str], bool]] = None,
):
    """
    Replaces LoRA weights on-the-fly using LoftQ technique.

    Workflow:
    1. Initialize SafetensorLoader
    2. For each quantized linear layer:
        a. Load original (non-quantized) weights
        b. Run _loftq_init_new to compute optimal LoRA
        c. Optionally validate with callback
        d. Replace or rollback based on callback result

    Callback Example:
    def validate_replacement(model, layer_name):
        logits_new = model(test_input)
        improvement = compute_metric(logits_new, logits_original)
        return improvement > threshold
    """
```

**Callback Mechanism**:
- Enables greedy optimization
- Validates each layer replacement
- Rolls back if performance degrades
- Allows incremental improvements across multiple runs

## Quantization Details

### Bit Packing (2/4-bit)

**Packing Example** (2-bit):
```
Original: [1, 0, 3, 2]
Binary:   [01, 00, 11, 10]
Packed:   10110001 (LIFO: last value in LSB)

Unpacking:
10110001 & 0b11 = 01 (first value)
10110001 >> 2 & 0b11 = 10 (second value)
...
```

**Storage Savings**:
- 2-bit: 16× compression vs float32
- 4-bit: 8× compression vs float32
- 8-bit: 4× compression vs float32

### Block-wise Quantization

**Why Block-wise?**
- Better accuracy than tensor-wise
- Handles varying magnitude across weight matrix
- Standard in modern quantization (GPTQ, AWQ)

**Block Size Trade-off**:
- Smaller blocks (e.g., 32): Higher accuracy, more overhead
- Larger blocks (e.g., 128): Lower accuracy, less overhead
- Default (64): Good balance

### Normal vs Uniform Quantization

**Normal Distribution**:
- Matches weight distribution better
- More levels near zero
- Better for typical neural network weights

**Uniform Distribution**:
- Simpler, faster
- Better for distributions with outliers
- Used in some quantization schemes

## Integration with bitsandbytes

### 4-bit Quantization

```python
if num_bits == 4 and is_bnb_4bit_available():
    qweight = bnb.nn.Params4bit(
        res.to("cpu"),
        requires_grad=False,
        compress_statistics=False,
        quant_type="nf4"
    ).to(compute_device)
    dequantized = bnb.functional.dequantize_4bit(qweight.data, qweight.quant_state)
```

**NF4 (Normal Float 4-bit)**:
- Optimized for normal distribution
- Used by QLoRA
- Best accuracy for 4-bit quantization

### 8-bit Quantization

```python
else:
    quantizer = NFQuantizer(num_bits=num_bits, device=device, method="normal", block_size=64)
    quantized_weight, max_abs, shape = quantizer.quantize_block(res)
    dequantized_weight = quantizer.dequantize_block(quantized_weight, max_abs, shape)
```

## Performance Characteristics

### Memory Usage

**During Initialization**:
- Original weight: O(m × n) float32
- Quantized weight: O(m × n / 8 × num_bits)
- LoRA factors: O((m + n) × r)
- Peak: Original + quantized (brief overlap)

**After Initialization**:
- Only quantized weight + LoRA factors remain
- Significant memory savings for large models

### Computational Cost

**Per Iteration**:
1. Quantization: O(m × n)
2. Dequantization: O(m × n)
3. SVD: O(min(m,n)² × max(m,n))
4. Total: Dominated by SVD

**Typical Settings**:
- `num_iter=1`: Single quantize-decompose cycle
- `num_iter=5`: Better quality, 5× slower

### Convergence

**Quantization Error Reduction**:
```
Iteration 1: 20% error → 15% error
Iteration 2: 15% error → 12% error
Iteration 3: 12% error → 10% error
...
```

Diminishing returns after 3-5 iterations.

## Use Cases

### 1. QLoRA Training

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM

# Load 4-bit quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# Create LoRA config with LoftQ
config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    init_lora_weights="loftq"  # Use LoftQ initialization
)
model = get_peft_model(model, config)
```

### 2. Post-hoc LoftQ Replacement

```python
# Load model with standard LoRA init
model = get_peft_model(quantized_model, lora_config)

# Replace with LoftQ-initialized weights
replace_lora_weights_loftq(
    model,
    model_path="meta-llama/Llama-2-7b-hf",
    adapter_name="default"
)
```

### 3. Validated Replacement (Greedy)

```python
def perplexity_callback(model, layer_name):
    """Only keep replacement if it improves perplexity."""
    ppl_before = compute_perplexity(model, validation_data)
    # Weights already replaced, compute new perplexity
    ppl_after = compute_perplexity(model, validation_data)
    return ppl_after < ppl_before

replace_lora_weights_loftq(
    model,
    model_path=model_path,
    callback=perplexity_callback
)
```

## Error Handling

### Common Errors

```python
# bitsandbytes not available
ValueError: "bitsandbytes is not available, please install it to use LoftQ."

# Unsupported bit width
ValueError: "Only support 2, 4, 8 bits quantization"

# Invalid iterations
ValueError: "Number of iterations must be greater than 0"

# Model not quantized
ValueError: "No bnb LoRA module found on the model"

# Weight shape mismatch
ValueError: "Only support 2D matrix, but your input has {ndim} dimensions."

# Block size mismatch
ValueError: "Weight with shape (m, n) is not dividable by block size {block_size}."
```

## Best Practices

1. **Iteration Count**:
   - Start with `num_iter=1` for speed
   - Increase to 3-5 if quality critical
   - Diminishing returns beyond 5

2. **Rank Selection**:
   - Higher rank: Better approximation, more parameters
   - Lower rank: Worse approximation, fewer parameters
   - LoftQ helps bridge the gap at lower ranks

3. **Validation**:
   - Use callback to validate improvements
   - Compare against baseline (standard init)
   - Monitor task-specific metrics

4. **Memory Management**:
   - Use `clear_device_cache()` between iterations
   - Load weights lazily with SafetensorLoader
   - Offload CPU if needed

5. **Numerical Stability**:
   - Use bfloat16/float32 for intermediate computations
   - Avoid float16 (can cause NaNs in SVD)
   - Check for NaNs after quantization

## Cross-References

- **Used By**: `peft.tuners.lora.LoraConfig` (init_lora_weights="loftq")
- **Related Papers**: [QLoRA](https://huggingface.co/papers/2305.14314), [LoftQ](https://huggingface.co/papers/2310.08659)
- **Dependencies**: `torch`, `bitsandbytes`, `safetensors`, `huggingface_hub`, `scipy`
- **See Also**: `peft.utils.incremental_pca` (low-rank decomposition)
