# src/peft/tuners/trainable_tokens/layer.py

## Overview
Implementation of TrainableTokensLayer, which wraps embedding layers to enable selective token training. This layer maintains separate trainable embeddings for specified tokens while keeping the rest of the embedding matrix frozen.

## Class: TrainableTokensLayer

Inherits from both `nn.Module` and `BaseTunerLayer`. This wrapper enables fine-grained control over which token embeddings are trainable.

### Class Attributes

- **adapter_layer_names**: `("trainable_tokens_delta",)` - Contains trainable token deltas
- **other_param_names**: `("token_indices", "trainable_tokens_original")` - Additional adapter-related parameters

---

### Initialization

**Parameters:**
- **base_layer**: The original embedding layer
- **adapter_name**: Name of the first adapter
- **token_indices**: List of token indices to make trainable
- **tied_adapter**: Optional reference to another TrainableTokensLayer for weight tying
- **kwargs**: Additional arguments

**Initialization Process:**

**If Not Tied:**
1. Create `trainable_tokens_delta` ParameterDict (stores trainable embeddings)
2. Create `trainable_tokens_original` BufferDict (stores original values for unmerging)
3. Create `token_indices` dict (maps adapter to its token indices)

**If Tied:**
1. Share all three data structures with the tied adapter
2. No separate storage - everything references tied adapter's storage
3. Ensures weight tying at parameter level

**Common Setup:**
4. Wrap tied_adapter in list (excludes from state dict)
5. Initialize empty merged_adapters list
6. Extract and store in_features, out_features

**Why Wrap in List?**
Wrapping `tied_adapter` in a list prevents it from being:
- Included in `.modules()` iteration
- Saved in state dict (would be duplicate)
- Treated as a submodule

---

### Core Methods

#### _collect_token_weights()
DeepSpeed Zero3-specific method for initializing trainable tokens in distributed setting.

**Parameters:**
- **weight**: Embedding weight tensor
- **rows**: Token indices to collect
- **embed_dim**: Embedding dimension

**Process:**
1. Determine source rank (rank 0)
2. Get current device
3. Use `gather_params_ctx` to temporarily gather sharded weights
4. If on source rank: Extract token embeddings
5. If on other ranks: Create empty tensor with correct shape/type
6. Broadcast token embeddings from source to all ranks

**Purpose:** In DeepSpeed Zero3, weights are sharded across ranks. This method ensures all ranks get the token embeddings they need for initialization.

**Current Limitation:** Only CUDA is implemented

#### update_layer()
Creates or updates an adapter with trainable tokens.

**Parameters:**
- **adapter_name**: Name for this adapter
- **kwargs**: Contains `token_indices`, `init_weights`, `tied_adapter`

**Process for Tied Adapters:**
- Simply return (tied adapters follow their parent)

**Process for Regular Adapters:**
1. Store token_indices for this adapter
2. Get initialization preference
3. Get base layer weight tensor
4. Determine embedding dimension
5. Get initial values:
   - If `init_weights=True` (default):
     - Check if DeepSpeed Zero3 active
     - If yes: Use `_collect_token_weights()`
     - If no: Direct indexing of weight matrix
   - If `init_weights=False`:
     - Create random tensor matching dtype/device
6. Store as trainable parameter in `trainable_tokens_delta`
7. Store original values in `trainable_tokens_original` buffer
8. Move to appropriate device

**Why Store Originals?**
Needed for unmerging - must restore original embeddings when adapter is removed.

#### _check_overlapping_tokens()
Validates that adapters don't have overlapping token indices.

**Parameters:**
- **adapter_names**: List of adapter names to check

**Process:**
1. Build set of all token indices across adapters
2. For each adapter:
   - Get its token index set
   - Check intersection with existing indices
   - If overlap found: Raise ValueError
   - Add to global set
3. Also checks already-merged adapters

**Why No Overlap?**
Multiple adapters updating same token would require merging strategy:
- Which adapter's value to use?
- How to combine updates?
- What about conflicts?

Current implementation keeps it simple: disjoint token sets only.

#### merge()
Merges adapter(s) into the base embedding layer.

**Parameters:**
- **safe_merge** (bool, default=False): Check for NaN/Inf after merging
- **adapter_names**: Optional list of adapters to merge

**Process:**
1. Validate adapters with `check_adapters_to_merge()`
2. Check for overlapping tokens
3. Start with base layer weight
4. For each adapter:
   - Get token indices as tensor
   - Get delta embeddings
   - Use `index_copy` to replace embeddings at indices
   - Optionally check for NaN/Inf
5. Update base layer weight
6. Add adapters to merged_adapters list

**Key Operation:** `merged.index_copy(dim=0, index=index, source=deltas)`
- Copies deltas into merged at specified indices
- In-place operation on dimension 0 (token dimension)
- Replaces entire embedding vectors

#### unmerge()
Reverses merge operation, restoring original embeddings.

**Process:**
1. Check if merged (warn if not)
2. While merged_adapters not empty:
   - Pop an adapter name
   - Get its token indices
   - Get original embeddings
   - Use `index_copy_` to restore originals
3. Clear merged_adapters list

**Difference from merge:** Uses `index_copy_` (in-place) instead of `index_copy` (out-of-place)

#### get_merged_weights()
Computes merged weights without modifying base layer.

**Parameters:**
- **active_adapters**: List of adapters to merge

**Process:**
1. Start with base layer weight
2. For each adapter:
   - Get token indices
   - Get delta embeddings
   - Copy into weight tensor
3. Return merged tensor (not nn.Parameter)

**Note:** Returns Tensor, not Parameter. This can cause issues with methods expecting Parameter type, but returning Parameter would create other problems (not a true model parameter).

#### forward_adapters()
Main forward logic for adapters.

**Parameters:**
- **x**: Input token IDs
- **active_adapters**: List of active adapters
- **args, kwargs**: Additional arguments

**Process:**

**If Disabled or No Active Adapters:**
1. Unmerge if currently merged
2. Use base_layer for forward

**If Merged:**
1. Use base_layer for forward (merged weights already in base)

**If Active Adapters:**
1. Check no overlapping tokens
2. Get merged weights from all active adapters
3. Determine layer type (Embedding or Linear)
4. **For Embedding:**
   - Use `F.embedding()` with merged weights
   - Apply embedding-specific parameters (padding_idx, max_norm, etc.)
   - Apply embed_scale if present (e.g., Gemma models)
5. **For Linear:**
   - Use `F.linear()` with merged weights
   - Used for tied weights with LM head
6. Return result

**Why Two Layer Types?**
- **Embedding**: Normal case, wrapping embedding layer
- **Linear**: Weight-tied case, wrapping LM head (which is Linear)

Must handle both to properly support weight tying.

#### forward()
Public forward method that delegates to `forward_adapters()`.

---

## Design Philosophy

### Delta Storage
Rather than storing full embeddings for selected tokens:
- **Advantage**: Clear semantics (these are the new embeddings)
- **Disadvantage**: Must store originals for unmerging
- **Trade-off**: Memory for clarity and correctness

Alternative would be:
- Store deltas to add to base embeddings
- No need for originals
- But breaks with resize_token_embeddings pattern

### Index-Copy Strategy
Using `index_copy` for merging:
- **Efficient**: Direct memory operation
- **Clear**: Explicitly replaces specified embeddings
- **Safe**: No risk of partial updates
- **Reversible**: Store originals to undo

### Tied Adapter Pattern
Sharing storage for tied adapters:
- **Memory Efficient**: No duplication
- **Correct**: Updates propagate automatically
- **Simple**: No synchronization logic needed
- **Clean**: State dict naturally excludes tied copy

---

## DeepSpeed Zero3 Support

### Why Special Handling?
In DeepSpeed Zero3:
- Weights sharded across ranks
- Each rank only has portion of parameters
- Need to gather for initialization
- Then broadcast to all ranks

### The Process
1. **Gather Phase**: Temporarily collect full weights on rank 0
2. **Extract Phase**: Get embeddings for specified tokens
3. **Broadcast Phase**: Share with all ranks
4. **Shard Phase**: Each rank keeps its portion

### Current Limitations
- Only CUDA supported
- Assumes standard distributed setup
- May need updates for custom configurations

---

## Special Case Handling

### Embedding Scaling
Some models (e.g., Gemma) scale embeddings in forward:
```python
embed_scale = self._get_embed_scale()
if embed_scale is not None:
    result = result * embed_scale.to(result.dtype)
```

TrainableTokensLayer preserves this:
- Detects if base layer has scaling
- Applies same scaling to adapter outputs
- Maintains model behavior

### LM Head Wrapping
Weight-tied models may have:
- Embedding layer wrapped normally
- LM head (Linear layer) also wrapped as tied adapter

Must handle both:
- Embedding: Use `F.embedding()`
- Linear: Use `F.linear()`

Layer type detection via `isinstance()` checks.

---

## Error Handling

### Overlapping Tokens
If adapters have overlapping token indices:
```
ValueError: Token indices of adapter {name} are already defined and would result in undefined merging behavior. Only disjunct token indices are currently supported.
```

**Prevention:** Ensure token indices are disjoint across adapters

### Unknown Layer Type
If wrapping unsupported layer:
```
ValueError: TrainableTokensLayer wraps an unknown layer type, maybe you are targeting the wrong layer?
```

**Prevention:** Only use with Embedding or Linear layers

---

## Integration with PEFT

### State Dict
Properly implements state dict handling:
- `trainable_tokens_delta`: Saved as parameters
- `trainable_tokens_original`: Saved as buffers
- `token_indices`: Saved as non-tensor attributes
- `_tied_adapter`: Excluded (wrapped in list)

### Parameter Counting
Parameters correctly counted:
- Only delta embeddings counted as trainable
- Original embeddings not counted (buffers)
- Tied adapters not double-counted

### Device Management
Proper device handling:
- Moves adapters to base layer device
- Handles cross-device operations
- Maintains consistency

---

## Memory Characteristics

For each adapter with N tokens and embedding dimension D:
- **Trainable Parameters**: N × D (delta embeddings)
- **Buffers**: N × D (original embeddings)
- **Metadata**: N integers (token indices)
- **Total**: ~2 × N × D

Compared to full embedding:
- **Full**: vocab_size × D
- **Trainable Tokens**: ~2 × N × D
- **Savings**: When N << vocab_size, massive savings

---

## Usage Patterns

### Single Adapter
```python
# One set of trainable tokens
layer.update_layer(
    "adapter1",
    token_indices=[100, 200, 300],
    init_weights=True,
)
```

### Multiple Adapters (Disjoint)
```python
# Two adapters, different tokens
layer.update_layer("adapter1", token_indices=[100, 200])
layer.update_layer("adapter2", token_indices=[300, 400])
# Forward can use both simultaneously
```

### Tied Adapters
```python
# Main adapter on embedding layer
main_layer.update_layer("adapter1", token_indices=[100, 200])

# Tied adapter on LM head
tied_layer.update_layer(
    "adapter1",
    tied_adapter=main_layer,  # Share storage
)
```

---

## Integration
This layer is created and injected by TrainableTokensModel. Users don't typically instantiate it directly. The model handles:
- Layer selection
- Adapter creation
- Weight tying setup
- Device management
