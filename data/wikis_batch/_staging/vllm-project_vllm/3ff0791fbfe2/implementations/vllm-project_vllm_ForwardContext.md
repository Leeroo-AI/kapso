# ForwardContext - Forward Pass Context Management

## Overview

**File:** `/tmp/praxium_repo_583nq7ea/vllm/forward_context.py` (358 lines)

**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)

**Purpose:** Manages global context state during model forward passes, enabling different model components to access shared execution metadata (attention metadata, batch descriptors, data parallelism info, CUDA graph mode) without explicit parameter passing.

## Core Architecture

### Data Structures

#### 1. BatchDescriptor (NamedTuple)

**Lines:** 28-57

Describes padded batches for CUDA graph dispatching:

```python
class BatchDescriptor(NamedTuple):
    num_tokens: int
    num_reqs: int | None = None     # None for PIECEWISE cudagraphs
    uniform: bool = False            # All requests same token count
    has_lora: bool = False          # Has active LoRA adapters

    def relax_for_mixed_batch_cudagraphs(self) -> "BatchDescriptor":
        # Returns descriptor compatible with PIECEWISE cudagraphs
        return BatchDescriptor(
            self.num_tokens, num_reqs=None, uniform=False, has_lora=self.has_lora
        )
```

**Purpose:** Minimal but complete batch characterization for CUDA graph cache key selection.

#### 2. DPMetadata (Dataclass)

**Lines:** 89-182

Manages token distribution across data parallel ranks:

```python
@dataclass
class DPMetadata:
    max_tokens_across_dp_cpu: torch.Tensor
    num_tokens_across_dp_cpu: torch.Tensor
    local_sizes: list[int] | None = None  # Set by chunked_sizes context

    @staticmethod
    def make(parallel_config, num_tokens, num_tokens_across_dp_cpu):
        # Validates consistency and computes max
        assert num_tokens_across_dp_cpu[dp_rank] == num_tokens
        max_tokens = torch.max(num_tokens_across_dp_cpu)
        return DPMetadata(max_tokens, num_tokens_across_dp_cpu)
```

**Key Methods:**

**chunked_sizes():** Context manager for chunked forward execution
```python
@contextmanager
def chunked_sizes(self, sequence_parallel_size, max_chunk_size_per_rank, chunk_idx):
    # Computes per-rank local token sizes for specific chunk
    self.local_sizes = _compute_chunked_local_num_tokens(
        self.num_tokens_across_dp_cpu,
        sequence_parallel_size,
        max_chunk_size_per_rank,
        chunk_idx
    )
    try:
        yield self.local_sizes
    finally:
        self.local_sizes = None
```

**sp_local_sizes():** Sequence parallel version without chunking
```python
@contextmanager
def sp_local_sizes(self, sequence_parallel_size):
    self.local_sizes = _compute_sp_num_tokens(
        self.num_tokens_across_dp_cpu, sequence_parallel_size
    )
    try:
        yield self.local_sizes
    finally:
        self.local_sizes = None
```

**cu_tokens_across_sp():** Cumulative token counts across sequence parallel ranks

#### 3. ForwardContext (Dataclass)

**Lines:** 185-212

Central context container:

```python
@dataclass
class ForwardContext:
    no_compile_layers: dict[str, Any]            # Static config
    attn_metadata: dict[str, AttentionMetadata]  # Per-layer attention metadata
    virtual_engine: int                           # For multi-engine setups
    dp_metadata: DPMetadata | None = None         # Data parallel metadata
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE
    batch_descriptor: BatchDescriptor | None = None
    ubatch_slices: UBatchSlices | None = None     # Microbatching info

    def __post_init__(self):
        assert self.cudagraph_runtime_mode.valid_runtime_modes()
```

### Helper Functions

#### Computing Sequence Parallel Token Counts

**Lines:** 60-86

```python
def _compute_sp_num_tokens(num_tokens_across_dp_cpu, sequence_parallel_size):
    # Ceiling division for even distribution
    sp_tokens = (num_tokens_across_dp_cpu + sequence_parallel_size - 1) // sequence_parallel_size
    # Repeat for each SP rank
    return sp_tokens.repeat_interleave(sequence_parallel_size).tolist()

def _compute_chunked_local_num_tokens(
    num_tokens_across_dp_cpu, sequence_parallel_size, max_num_tokens, chunk_idx
):
    sp_tokens = _compute_sp_num_tokens(num_tokens_across_dp_cpu, sequence_parallel_size)
    local_size = [-1] * len(sp_tokens)
    for i in range(len(sp_tokens)):
        # Compute chunk with bounds checking
        local_size[i] = min(max_num_tokens, sp_tokens[i] - (max_num_tokens * chunk_idx))
        if local_size[i] <= 0:
            local_size[i] = 1  # Ensure lockstep execution
    return local_size
```

### Global Context Management

**Lines:** 214-263

```python
_forward_context: ForwardContext | None = None

def get_forward_context() -> ForwardContext:
    assert _forward_context is not None, "Forward context not set"
    return _forward_context

def is_forward_context_available() -> bool:
    return _forward_context is not None

def create_forward_context(...) -> ForwardContext:
    return ForwardContext(
        no_compile_layers=vllm_config.compilation_config.static_forward_context,
        virtual_engine=virtual_engine,
        attn_metadata=attn_metadata,
        dp_metadata=dp_metadata,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor,
        ubatch_slices=ubatch_slices,
    )

@contextmanager
def override_forward_context(forward_context: ForwardContext | None):
    global _forward_context
    prev_context = _forward_context
    _forward_context = forward_context
    try:
        yield
    finally:
        _forward_context = prev_context
```

### Main Context Manager

**Lines:** 265-358

```python
@contextmanager
def set_forward_context(
    attn_metadata, vllm_config, virtual_engine=0,
    num_tokens=None, num_tokens_across_dp=None,
    cudagraph_runtime_mode=CUDAGraphMode.NONE,
    batch_descriptor=None, ubatch_slices=None
):
    # Start timing for batch size logging
    if track_batchsize and attn_metadata is not None:
        forward_start_time = time.perf_counter()

    # Initialize DP metadata if needed
    dp_metadata = None
    if vllm_config.parallel_config.data_parallel_size > 1:
        if num_tokens_across_dp is None:
            _, num_tokens_across_dp, _ = coordinate_batch_across_dp(
                num_tokens_unpadded=num_tokens,
                parallel_config=vllm_config.parallel_config,
                allow_microbatching=False,
                allow_dp_padding=False,
            )
        dp_metadata = DPMetadata.make(
            vllm_config.parallel_config, num_tokens, num_tokens_across_dp
        )

    # Create batch descriptor if CUDA graphs enabled
    if cudagraph_runtime_mode != CUDAGraphMode.NONE and num_tokens is not None:
        batch_descriptor = batch_descriptor or BatchDescriptor(num_tokens=num_tokens)

    forward_context = create_forward_context(...)

    try:
        with override_forward_context(forward_context):
            yield
    finally:
        # Log batch size statistics if enabled
        if track_batchsize:
            synchronize()
            now = time.perf_counter()
            batchsize_forward_time[batchsize].append((now - forward_start_time) * 1000)

            if now - last_logging_time > batchsize_logging_interval:
                # Compute and log median times per batch size
                forward_stats = []
                for bs, times in batchsize_forward_time.items():
                    if len(times) > 1:
                        median = torch.quantile(torch.tensor(times), q=0.5).item()
                        forward_stats.append((bs, len(times), round(median, 2)))
                logger.info("Batchsize forward time stats: %s", forward_stats)
```

## Implementation Details

### Batch Size Logging

**Lines:** 21-25, 326-358

Optional performance tracking controlled by `VLLM_LOG_BATCHSIZE_INTERVAL`:

```python
track_batchsize: bool = envs.VLLM_LOG_BATCHSIZE_INTERVAL >= 0
last_logging_time: float = 0
forward_start_time: float = 0
batchsize_logging_interval: float = envs.VLLM_LOG_BATCHSIZE_INTERVAL
batchsize_forward_time: defaultdict = defaultdict(list)
```

### Data Parallel Coordination

The `DPMetadata` class ensures lockstep execution across DP ranks:
- All ranks process consistent token counts (with padding if needed)
- Chunked execution maintains synchronization even when some ranks finish early
- Sequence parallel distribution accounts for uneven token counts

### CUDA Graph Integration

The `BatchDescriptor` enables efficient CUDA graph caching:
- **FULL mode:** Exact match on all fields (num_tokens, num_reqs, uniform, has_lora)
- **PIECEWISE mode:** Relaxed matching (only num_tokens and has_lora)
- Descriptor automatically created if CUDA graphs enabled

## Usage Patterns

### Basic Usage

```python
with set_forward_context(
    attn_metadata=attn_meta,
    vllm_config=config,
    virtual_engine=0,
    num_tokens=batch_size
):
    output = model.forward(input_ids)  # Components access context via get_forward_context()
```

### With Data Parallelism

```python
with set_forward_context(
    attn_metadata=attn_meta,
    vllm_config=config,
    num_tokens=local_batch_size,
    num_tokens_across_dp=all_batch_sizes  # Tensor of all DP ranks' sizes
):
    ctx = get_forward_context()
    # Access DP metadata
    max_tokens = ctx.dp_metadata.max_tokens_across_dp_cpu
```

### With Chunked Execution

```python
ctx = get_forward_context()
with ctx.dp_metadata.chunked_sizes(sp_size=4, max_chunk_size=1024, chunk_idx=0):
    local_sizes = ctx.dp_metadata.get_chunk_sizes_across_dp_rank()
    # Process first chunk with local_sizes
```

### Accessing Context in Layers

```python
def attention_forward(self, hidden_states):
    ctx = get_forward_context()
    attn_metadata = ctx.attn_metadata[self.layer_name]
    # Use attention metadata for current layer
    return self._attention_impl(hidden_states, attn_metadata)
```

## Integration Points

### vllm.attention

Retrieves layer-specific `AttentionMetadata` from context.

### vllm.v1.worker.dp_utils

`coordinate_batch_across_dp()` initializes DP metadata.

### vllm.config

- `VllmConfig`: Provides compilation config and parallel config
- `CUDAGraphMode`: Enum for CUDA graph modes (NONE/FULL/PIECEWISE)

### vllm.v1.worker.ubatch_utils

`UBatchSlices` for microbatching support.

## Performance Implications

### Benefits

1. **Zero Parameter Overhead:** No need to pass metadata through all layers
2. **CUDA Graph Efficiency:** Batch descriptors enable optimal graph caching
3. **DP Synchronization:** Ensures efficient collective operations
4. **Observability:** Batch size logging provides performance insights

### Considerations

1. **Global State:** Thread-local storage would be needed for multi-threaded inference
2. **Memory:** Stores complete attention metadata for all layers
3. **Synchronization:** Batch logging adds sync point at end of forward pass

## Related Components

- **vllm/envs.py:** Environment variable configuration
- **vllm/logger.py:** Logging infrastructure
- **vllm/attention/backends/abstract.py:** AttentionMetadata definition
- **vllm/config.py:** Configuration classes
- **Model Implementations:** All models access context during forward pass

## Technical Significance

This module is fundamental to vLLM's architecture:
- **Simplifies Layer Implementation:** Layers don't need complex parameter signatures
- **Enables Advanced Features:** DP, chunking, CUDA graphs all depend on context
- **Type Safety:** Strong typing ensures correct context usage
- **Extensibility:** Easy to add new context fields as features evolve

The context management pattern trades global state for code simplicity, which is acceptable in the single-threaded inference server model.
