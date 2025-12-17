# Implementation: incremental_pca.py

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/utils/incremental_pca.py`
- **Size**: 338 lines
- **Module**: `peft.utils.incremental_pca`
- **Description**: Memory-efficient incremental PCA implementation with GPU acceleration

## Overview

This module provides a PyTorch-based Incremental Principal Component Analysis (IPCA) implementation that enables PCA computation on datasets too large to fit in memory. It processes data in batches, updating principal components incrementally, and leverages GPU acceleration for efficient computation. The implementation is adapted from scikit-learn's IncrementalPCA but optimized for PyTorch tensors and CUDA operations.

## Core Class: IncrementalPCA

### Initialization Parameters

```python
def __init__(
    self,
    n_components: Optional[int] = None,      # Number of components to keep
    copy: Optional[bool] = True,             # Whether to copy input data
    batch_size: Optional[int] = None,        # Samples per batch (auto: 5 * n_features)
    svd_driver: Optional[str] = None,        # cuSOLVER method for torch.linalg.svd
    lowrank: bool = False,                   # Use torch.svd_lowrank for speed
    lowrank_q: Optional[int] = None,         # Approximation quality (default: 2 * n_components)
    lowrank_niter: int = 4,                  # Subspace iterations for lowrank
    lowrank_seed: Optional[int] = None,      # Seed for lowrank reproducibility
)
```

### Key Attributes

**State Variables**:
- `n_features_`: Number of features in the data
- `n_samples_seen_`: Total samples processed (tensor)
- `components_`: Principal components (n_components, n_features)
- `singular_values_`: Singular values corresponding to components
- `mean_`: Feature-wise mean
- `var_`: Feature-wise variance
- `explained_variance_`: Variance explained by each component
- `explained_variance_ratio_`: Proportion of total variance explained
- `noise_variance_`: Estimated noise variance

## Core Methods

### Data Validation

**_validate_data(X: torch.Tensor) → torch.Tensor**
```python
def _validate_data(self, X) -> torch.Tensor:
    """
    Validates and converts input data to appropriate tensor format.

    Checks:
    - Converts to torch.Tensor if needed
    - Ensures float32 or float64 dtype
    - Validates n_components <= n_features
    - Validates n_components <= batch_size
    - Optionally copies data
    """
```

### Incremental Statistics

**_incremental_mean_and_var() → tuple[Tensor, Tensor, Tensor]**
```python
@staticmethod
def _incremental_mean_and_var(
    X, last_mean, last_variance, last_sample_count
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes incremental mean and variance using Welford's online algorithm.

    Algorithm:
    1. Compute new batch statistics
    2. Combine with previous statistics using weighted averaging
    3. Apply numerical stability corrections

    Returns: (updated_mean, updated_variance, total_sample_count)
    """
```

**Key Features**:
- **Numerically Stable**: Uses correction factor for variance computation
- **Memory Efficient**: O(n_features) space complexity
- **Parallel Computation**: Leverages PyTorch's vectorized operations

### SVD Computation

**Two SVD Backends**:

1. **Full SVD** (default, accurate):
```python
def _svd_fn_full(self, X):
    return torch.linalg.svd(X, full_matrices=False, driver=self.svd_driver)
```

2. **Low-rank SVD** (faster, approximate):
```python
def _svd_fn_lowrank(self, X):
    with torch.random.fork_rng(enabled=seed_enabled):
        if seed_enabled:
            torch.manual_seed(self.lowrank_seed)
        U, S, V = torch.svd_lowrank(X, q=self.lowrank_q, niter=self.lowrank_niter)
        return U, S, V.mH  # conjugate transpose
```

**Trade-offs**:
- Full SVD: Exact, slower, O(min(m,n)²·max(m,n)) complexity
- Low-rank: Approximate, faster, O(n_components·m·n) complexity

### Sign Flipping for Determinism

**_svd_flip() → tuple[Tensor, Tensor]**
```python
@staticmethod
def _svd_flip(u, v, u_based_decision=True):
    """
    Ensures deterministic output by adjusting singular vector signs.

    Strategy:
    - Find maximum absolute value in each column/row
    - Set sign based on the sign of that maximum
    - Apply consistent sign to both U and V
    """
```

## Training Methods

### Full Dataset Fitting

**fit(X, check_input=True) → IncrementalPCA**
```python
def fit(self, X, check_input=True):
    """
    Fits the model with data X using minibatches.

    Process:
    1. Validate input
    2. Determine batch_size (default: 5 * n_features)
    3. Generate batches
    4. Call partial_fit for each batch
    """
```

### Incremental Fitting

**partial_fit(X, check_input=True) → IncrementalPCA**
```python
def partial_fit(self, X, check_input=True):
    """
    Incrementally fits the model with batch data.

    Algorithm:
    1. Update mean and variance incrementally
    2. Center the new batch
    3. If not first pass: stack with previous components
    4. Perform SVD on stacked matrix
    5. Update components and statistics
    6. Retain top n_components
    """
```

**Key Steps**:

**First Pass** (no existing components):
```python
X -= col_mean  # Simply center
```

**Subsequent Passes** (existing components):
```python
# Mean correction
mean_correction = sqrt((n_old / n_total) * n_new) * (mean_old - mean_batch)

# Stack previous components, new data, and correction
X_stacked = vstack([
    singular_values.view(-1, 1) * components,  # Previous components
    X - col_batch_mean,                        # New centered data
    mean_correction                            # Correction term
])

# SVD on stacked matrix updates components
U, S, Vt = svd(X_stacked)
```

### Transformation

**transform(X) → torch.Tensor**
```python
def transform(self, X) -> torch.Tensor:
    """
    Projects data onto principal components.

    Operation: (X - mean) @ components.T

    Shape: (n_samples, n_features) → (n_samples, n_components)
    """
```

## Utility Methods

### Batch Generation

**gen_batches(n, batch_size, min_batch_size=0)**
```python
@staticmethod
def gen_batches(n: int, batch_size: int, min_batch_size: int = 0):
    """
    Generator creating slices of batch_size elements.

    Features:
    - Last batch may be smaller
    - Skip batches smaller than min_batch_size
    - Memory efficient (generator pattern)
    """
```

## Algorithm Details

### Incremental PCA Mathematics

**Update Equation**:
```
X_stacked = [sqrt(λ_prev) * V_prev]  # Previous components scaled by singular values
            [X_new - μ_batch]          # New centered data
            [sqrt((n_old/n_total)*n_new) * (μ_old - μ_batch)]  # Mean correction
```

**Why This Works**:
1. Previous components contain information from past data
2. Scaling by singular values reconstructs approximate covariance
3. Mean correction accounts for distribution shift
4. SVD of stacked matrix merges old and new information

### Variance Computation

**Welford's Algorithm** (numerically stable):
```python
# Batch variance with correction
temp = X - T  # T = batch_mean
correction = temp.sum(dim=0).square() / n_batch
unnorm_var_new = temp.square().sum(dim=0) - correction

# Combine with old variance
unnorm_var_updated = unnorm_var_old + unnorm_var_new +
                     (n_old/n_new)/n_total * (sum_old/count_old - sum_new).square()
```

## Performance Characteristics

### Memory Usage

- **Per Batch**: O(batch_size × n_features)
- **State**: O(n_components × n_features + n_features)
- **Total**: Independent of total dataset size (memory-efficient!)

### Computational Complexity

**Per partial_fit Call**:
- Centering: O(batch_size × n_features)
- SVD: O(min(batch_size, n_components)² × n_features)
- Total per sample: O(n_features × n_components)

**Full SVD**: O(min(m,n)²·max(m,n))
**Low-rank SVD**: O(q·m·n·niter), where q ≈ 2 × n_components

### GPU Acceleration

**cuSOLVER Integration**:
- `svd_driver`: Uses CUDA-optimized SVD when available
- Options: `gesvd`, `gesvdj`, `gesvda`
- Automatic device handling (works on CPU too)

## Use Cases in PEFT

### Adaptive Rank Allocation

**Example: AdaLoRA SVD Budget Allocation**
```python
ipca = IncrementalPCA(n_components=target_rank)
for batch in data_loader:
    activations = model.get_activations(batch)
    ipca.partial_fit(activations)

# Use explained variance ratio to determine per-layer ranks
layer_ranks = allocate_based_on_variance(ipca.explained_variance_ratio_)
```

### Initialization

**Example: Low-rank Initialization from Data**
```python
ipca = IncrementalPCA(n_components=r)
ipca.fit(weight_matrix)

# Initialize LoRA matrices with principal components
lora_A = ipca.components_.T  # (in_features, r)
lora_B = torch.zeros(r, out_features)  # Zero init for zero-init property
```

## Configuration Examples

### High Accuracy
```python
ipca = IncrementalPCA(
    n_components=64,
    batch_size=1024,
    lowrank=False,
    svd_driver="gesvdj"  # Most accurate cuSOLVER method
)
```

### High Speed
```python
ipca = IncrementalPCA(
    n_components=64,
    batch_size=1024,
    lowrank=True,
    lowrank_q=128,       # 2x oversampling
    lowrank_niter=2,     # Fewer iterations
    lowrank_seed=42      # For reproducibility
)
```

### Memory Constrained
```python
ipca = IncrementalPCA(
    n_components=32,
    batch_size=256,      # Small batches
    copy=False,          # Modify input in-place
    lowrank=True
)
```

## Error Handling

### Validation Errors

```python
# n_components too large
ValueError: "n_components=128 invalid for n_features=64"

# n_components > batch_size
ValueError: "n_components=64 must be less or equal to the batch number of samples 32"

# Feature mismatch
ValueError: "Number of features of the new batch does not match first batch"

# lowrank_q too small
ValueError: "lowrank_q must be greater than or equal to n_components"
```

## Differences from scikit-learn

1. **PyTorch Tensors**: Operates on torch.Tensor instead of numpy arrays
2. **GPU Support**: Native CUDA acceleration
3. **Low-rank Option**: torch.svd_lowrank for faster computation
4. **Device Handling**: Automatic device placement and dtype management
5. **Batch Generator**: Simplified generator pattern

## Best Practices

1. **Batch Size Selection**:
   - Default (5 × n_features) works well in most cases
   - Larger batches: More accurate, more memory
   - Smaller batches: Less accurate, less memory

2. **Component Selection**:
   - Start with n_components ≈ 0.95 explained variance
   - Use scree plot (singular values) to guide selection

3. **Low-rank Mode**:
   - Use when n_components << n_features
   - Set lowrank_q = 2 × n_components for good approximation
   - Increase lowrank_niter if accuracy critical

4. **Numerical Stability**:
   - Use float64 for high-precision requirements
   - Check condition number of covariance matrix
   - Monitor explained_variance_ratio for quality

## Cross-References

- **Used By**: `peft.tuners.adalora` (rank allocation), `peft.utils.loftq_utils` (initialization)
- **Related**: scikit-learn's IncrementalPCA (original inspiration)
- **Dependencies**: `torch`, `torch.linalg`, `torch.random`
