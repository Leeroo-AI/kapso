# PyTorch Utils - HuggingFace Transformers

## Metadata

| Property | Value |
|----------|-------|
| Source | `src/transformers/pytorch_utils.py` |
| Repository | huggingface/transformers |
| Commit Hash | f9f6619c2cf7 |
| Domain | Machine Learning / Deep Learning |
| Primary Language | Python |
| License | Apache License 2.0 |
| Last Updated | 2025-12-18 |

## Overview

The `pytorch_utils.py` module provides essential utility functions and custom layers for PyTorch operations within the Transformers library. It includes version-aware utilities, tensor operations, custom neural network layers, and compatibility functions that handle differences across PyTorch versions and device types (CPU, CUDA, MPS, XLA).

## Description

This module serves as a compatibility and utility layer that abstracts away PyTorch version differences and provides common operations used throughout the Transformers library. Key functionalities include:

1. **Version Detection**: Checks for specific PyTorch versions to enable version-specific features and workarounds.

2. **Custom Layers**: Implements specialized layers like `Conv1D` used in GPT models, which behaves like a linear layer with transposed weights.

3. **Tensor Utilities**: Provides functions for tensor operations including pruning, chunking, meshgrid wrapping, and tensor storage identification.

4. **Device Compatibility**: Handles device-specific operations for CPU, CUDA, MPS (Apple Silicon), and XLA (TPU) with appropriate fallbacks.

5. **Performance Optimization**: Includes utilities for gradient computation, memory-efficient chunked forward passes, and compile-compatible LRU caching.

The module is essential for maintaining cross-version compatibility and providing optimized implementations of common operations used in transformer models.

### Key Features

- PyTorch version detection and compatibility flags
- Custom Conv1D layer for GPT-style models
- Linear layer pruning for model compression
- Memory-efficient chunked forward propagation
- Tensor storage identification for memory management
- MPS-friendly tensor operations with fallbacks
- Compile-compatible method caching

## Usage

### Basic Usage

```python
from transformers.pytorch_utils import Conv1D, prune_linear_layer, apply_chunking_to_forward

# Use the Conv1D layer (GPT-style)
conv_layer = Conv1D(nf=768, nx=3072)
output = conv_layer(input_tensor)

# Prune a linear layer to remove specific neurons
layer = nn.Linear(1024, 768)
indices_to_keep = torch.tensor([0, 1, 2, 5, 10])
pruned_layer = prune_linear_layer(layer, indices_to_keep, dim=0)

# Apply chunking for memory-efficient processing
def forward_chunk(hidden_states):
    return model.decoder(hidden_states)

chunked_output = apply_chunking_to_forward(
    forward_chunk,
    chunk_size=256,
    chunk_dim=1,
    hidden_states
)
```

### Version Checking

```python
from transformers.pytorch_utils import (
    is_torch_greater_or_equal_than_2_8,
    is_torch_greater_or_equal_than_2_6,
    is_torch_greater_or_equal_than_2_4
)

# Use version-specific features
if is_torch_greater_or_equal_than_2_6:
    # Use features available in PyTorch 2.6+
    result = torch.some_new_function()
else:
    # Fallback for older versions
    result = legacy_implementation()
```

### Tensor Storage Identification

```python
from transformers.pytorch_utils import id_tensor_storage

# Get unique identifier for tensor storage
device, storage_id, size = id_tensor_storage(tensor)
print(f"Tensor on {device} with storage ID {storage_id} and size {size}")

# Use for deduplication or memory tracking
tensor_map = {}
for tensor in tensor_list:
    storage_key = id_tensor_storage(tensor)
    if storage_key not in tensor_map:
        tensor_map[storage_key] = tensor
```

### MPS-Friendly Operations

```python
from transformers.pytorch_utils import isin_mps_friendly

# Safe torch.isin replacement that works on MPS devices
elements = torch.tensor([1, 2, 3, 4, 5]).to("mps")
test_elements = torch.tensor([2, 4])

# This works correctly on MPS even with older PyTorch versions
mask = isin_mps_friendly(elements, test_elements)
# Result: tensor([False, True, False, True, False])
```

## Code Reference

### Main Functions

#### prune_linear_layer

```python
def prune_linear_layer(
    layer: nn.Linear,
    index: torch.LongTensor,
    dim: int = 0
) -> nn.Linear:
    """
    Prune a linear layer to keep only entries in index.

    Args:
        layer: The layer to prune
        index: The indices to keep in the layer
        dim: The dimension on which to keep the indices (0 for output, 1 for input)

    Returns:
        The pruned layer as a new layer with requires_grad=True
    """
```

#### apply_chunking_to_forward

```python
def apply_chunking_to_forward(
    forward_fn: Callable[..., torch.Tensor],
    chunk_size: int,
    chunk_dim: int,
    *input_tensors,
) -> torch.Tensor:
    """
    Chunks input tensors and applies forward_fn to each chunk independently.

    Args:
        forward_fn: The forward function of the model
        chunk_size: The chunk size of a chunked tensor
        chunk_dim: The dimension over which to chunk
        input_tensors: The input tensors to be chunked

    Returns:
        A tensor with the same shape as forward_fn would have given if applied directly
    """
```

#### id_tensor_storage

```python
def id_tensor_storage(tensor: torch.Tensor) -> tuple[torch.device, int, int]:
    """
    Unique identifier to a tensor storage.

    Args:
        tensor: The tensor to identify

    Returns:
        Tuple of (device, unique_id, storage_size)
    """
```

#### isin_mps_friendly

```python
def isin_mps_friendly(
    elements: torch.Tensor,
    test_elements: torch.Tensor | int
) -> torch.Tensor:
    """
    Same as torch.isin but MPS-friendly for PyTorch <= 2.3.

    Args:
        elements: Input elements
        test_elements: The elements to check against

    Returns:
        A boolean tensor indicating which elements are in test_elements
    """
```

#### softmax_backward_data

```python
def softmax_backward_data(
    parent,
    grad_output,
    output
):
    """
    Calls internal PyTorch _softmax_backward_data method with version-appropriate arguments.

    Args:
        parent: The parent module
        grad_output: Gradient of the output
        output: The output tensor

    Returns:
        Gradient with respect to the input
    """
```

#### compile_compatible_method_lru_cache

```python
def compile_compatible_method_lru_cache(*lru_args, **lru_kwargs):
    """
    LRU cache decorator that disables caching during torchdynamo compilation.

    Returns:
        Decorated function with compile-compatible caching
    """
```

#### meshgrid

```python
def meshgrid(
    *tensors: torch.Tensor | list[torch.Tensor],
    indexing: str | None = None
) -> tuple[torch.Tensor, ...]:
    """
    Wrapper around torch.meshgrid to avoid deprecation warnings.

    Args:
        tensors: Input tensors
        indexing: Indexing mode ('xy' or 'ij')

    Returns:
        Tuple of meshgrid tensors
    """
```

### Custom Layers

#### Conv1D

```python
class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT.
    Works like a linear layer but with transposed weights.

    Args:
        nf: The number of output features
        nx: The number of input features
    """

    def __init__(self, nf, nx):
        """Initialize Conv1D layer."""

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (..., nx)

        Returns:
            Output tensor of shape (..., nf)
        """
```

### Constants

```python
ALL_LAYERNORM_LAYERS = [nn.LayerNorm]

# Version flags
is_torch_greater_or_equal_than_2_8: bool
is_torch_greater_or_equal_than_2_6: bool
is_torch_greater_or_equal_than_2_4: bool
is_torch_greater_or_equal_than_2_3: bool
is_torch_greater_or_equal_than_2_2: bool
is_torch_greater_or_equal_than_2_1: bool
is_torch_greater_or_equal_than_2_0: bool
is_torch_greater_or_equal_than_1_13: bool
is_torch_greater_or_equal_than_1_12: bool
```

### Imports

```python
from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import lru_cache, wraps

import torch
from safetensors.torch import storage_ptr, storage_size
from torch import nn

from .utils import (
    is_torch_greater_or_equal,
    is_torch_xla_available,
    is_torchdynamo_compiling,
    logging,
)
```

## I/O Contracts

### prune_linear_layer

#### Inputs
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| layer | nn.Linear | The layer to prune | Required |
| index | torch.LongTensor | Indices to keep | Required |
| dim | int | Dimension to prune (0=output, 1=input) | 0 |

#### Outputs
| Field | Type | Description |
|-------|------|-------------|
| new_layer | nn.Linear | Pruned layer with requires_grad=True |

### apply_chunking_to_forward

#### Inputs
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| forward_fn | Callable | Function to apply to chunks | Required |
| chunk_size | int | Size of each chunk | Required |
| chunk_dim | int | Dimension to chunk along | Required |
| input_tensors | tuple | Tensors to be chunked | Required |

#### Outputs
| Field | Type | Description |
|-------|------|-------------|
| result | torch.Tensor | Concatenated output of all chunks |

### id_tensor_storage

#### Inputs
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| tensor | torch.Tensor | Tensor to identify | Required |

#### Outputs
| Field | Type | Description |
|-------|------|-------------|
| device | torch.device | Device where tensor is stored |
| unique_id | int | Unique storage pointer/identifier |
| size | int | Size of storage in bytes |

### isin_mps_friendly

#### Inputs
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| elements | torch.Tensor | Input elements to check | Required |
| test_elements | torch.Tensor or int | Elements to check against | Required |

#### Outputs
| Field | Type | Description |
|-------|------|-------------|
| mask | torch.Tensor | Boolean mask of same shape as elements |

### Conv1D

#### Inputs
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| x | torch.Tensor | Input tensor of shape (..., nx) | Required |

#### Outputs
| Field | Type | Description |
|-------|------|-------------|
| output | torch.Tensor | Output tensor of shape (..., nf) |

## Usage Examples

### Example 1: Pruning Attention Heads

```python
from transformers.pytorch_utils import prune_linear_layer
import torch.nn as nn

class PrunableAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

    def prune_heads(self, heads_to_prune):
        """Prune specific attention heads."""
        if len(heads_to_prune) == 0:
            return

        # Keep only the heads we're not pruning
        heads_to_keep = [h for h in range(self.num_heads) if h not in heads_to_prune]

        # Calculate indices for the features corresponding to kept heads
        index = torch.tensor([
            i for h in heads_to_keep
            for i in range(h * self.head_dim, (h + 1) * self.head_dim)
        ])

        # Prune query, key, value layers
        self.query = prune_linear_layer(self.query, index, dim=0)
        self.key = prune_linear_layer(self.key, index, dim=0)
        self.value = prune_linear_layer(self.value, index, dim=0)

        # Prune output layer
        self.out = prune_linear_layer(self.out, index, dim=1)

        # Update number of heads
        self.num_heads = len(heads_to_keep)

# Usage
attention = PrunableAttention(hidden_size=768, num_heads=12)
attention.prune_heads([0, 5, 11])  # Prune heads 0, 5, and 11
print(f"Remaining heads: {attention.num_heads}")  # Output: 9
```

### Example 2: Memory-Efficient Processing with Chunking

```python
from transformers.pytorch_utils import apply_chunking_to_forward

class MemoryEfficientFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.activation = nn.GELU()
        self.chunk_size = 256  # Process 256 tokens at a time

    def forward_chunk(self, hidden_states):
        """Forward pass for a chunk of data."""
        intermediate = self.activation(self.dense1(hidden_states))
        output = self.dense2(intermediate)
        return output

    def forward(self, hidden_states):
        """Forward pass with automatic chunking."""
        # Process in chunks to save memory
        return apply_chunking_to_forward(
            self.forward_chunk,
            self.chunk_size,
            1,  # chunk along sequence dimension
            hidden_states
        )

# Usage
ffn = MemoryEfficientFFN(hidden_size=768, intermediate_size=3072)
large_input = torch.randn(4, 2048, 768)  # Large sequence
output = ffn(large_input)  # Processed in chunks of 256 tokens
```

### Example 3: Conv1D Layer for GPT Models

```python
from transformers.pytorch_utils import Conv1D
import torch.nn as nn

class GPTStyleMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        # Conv1D is used in GPT models instead of Linear
        self.c_fc = Conv1D(intermediate_size, hidden_size)
        self.c_proj = Conv1D(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: (batch, seq_len, hidden_size)
        h = self.activation(self.c_fc(x))
        h = self.c_proj(h)
        h = self.dropout(h)
        return h

# Usage
mlp = GPTStyleMLP(hidden_size=768, intermediate_size=3072)
input_tensor = torch.randn(2, 128, 768)
output = mlp(input_tensor)
print(f"Output shape: {output.shape}")  # (2, 128, 768)
```

### Example 4: Version-Aware Feature Usage

```python
from transformers.pytorch_utils import (
    is_torch_greater_or_equal_than_2_4,
    is_torch_greater_or_equal_than_2_0,
    isin_mps_friendly
)
import torch

class VersionAwareModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_flex_attention = is_torch_greater_or_equal_than_2_4
        self.use_compile = is_torch_greater_or_equal_than_2_0

    def filter_tokens(self, input_ids, special_tokens):
        """Filter out special tokens in a device-agnostic way."""
        # Use MPS-friendly version which handles older PyTorch versions
        mask = ~isin_mps_friendly(input_ids, special_tokens)
        return input_ids[mask]

    def forward(self, x):
        if self.use_flex_attention:
            # Use newer attention implementation
            output = self.flex_attention(x)
        else:
            # Fallback to standard attention
            output = self.standard_attention(x)
        return output

# Compile if supported
model = VersionAwareModel()
if model.use_compile:
    model = torch.compile(model)
```

### Example 5: Tensor Storage Management

```python
from transformers.pytorch_utils import id_tensor_storage

class TensorCache:
    def __init__(self):
        self.storage_cache = {}

    def add_tensor(self, name, tensor):
        """Add tensor to cache, avoiding duplicates."""
        storage_id = id_tensor_storage(tensor)

        if storage_id in self.storage_cache:
            print(f"Tensor {name} shares storage with {self.storage_cache[storage_id]}")
        else:
            self.storage_cache[storage_id] = name
            print(f"Added tensor {name} with unique storage")

    def memory_usage(self):
        """Calculate total unique memory usage."""
        total_bytes = sum(size for (_, _, size) in self.storage_cache.keys())
        return total_bytes / (1024 ** 2)  # Convert to MB

# Usage
cache = TensorCache()
tensor1 = torch.randn(1000, 1000)
tensor2 = tensor1.view(-1)  # Shares storage with tensor1
tensor3 = torch.randn(500, 500)  # Different storage

cache.add_tensor("tensor1", tensor1)
cache.add_tensor("tensor2", tensor2)  # Will detect shared storage
cache.add_tensor("tensor3", tensor3)

print(f"Total memory: {cache.memory_usage():.2f} MB")
```

### Example 6: Compile-Compatible Caching

```python
from transformers.pytorch_utils import compile_compatible_method_lru_cache
import torch

class CachedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 768)

    @compile_compatible_method_lru_cache(maxsize=128)
    def compute_position_embeddings(self, seq_len, device):
        """Cache position embeddings but disable during compilation."""
        positions = torch.arange(seq_len, device=device)
        # Expensive computation that benefits from caching
        return self.expensive_embedding_function(positions)

    def forward(self, x):
        seq_len = x.size(1)
        pos_emb = self.compute_position_embeddings(seq_len, x.device)
        return self.linear(x) + pos_emb

# The cache works normally
model = CachedModel()
out1 = model(torch.randn(2, 128, 768))  # Computes embeddings
out2 = model(torch.randn(2, 128, 768))  # Uses cached embeddings

# But during torch.compile, caching is disabled to avoid issues
compiled_model = torch.compile(model)
out3 = compiled_model(torch.randn(2, 128, 768))  # Cache bypassed during compilation
```

## Related Pages

- (To be populated)
