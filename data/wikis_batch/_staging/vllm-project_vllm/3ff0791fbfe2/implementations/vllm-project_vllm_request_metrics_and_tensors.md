# Implementation: Request Metrics and Intermediate Tensors

**File:** `/tmp/praxium_repo_583nq7ea/vllm/sequence.py` (98 lines)
**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)

## Overview

The `sequence.py` module (partial, 98 lines) defines fundamental data structures for request tracking and pipeline communication in vLLM. It provides `RequestMetrics` for comprehensive timing information and `IntermediateTensors` for passing hidden states between pipeline stages.

**Key Components:**
- `RequestMetrics`: Comprehensive timing data for request lifecycle
- `IntermediateTensors`: Container for pipeline stage intermediate outputs
- Constants for token handling: `VLLM_TOKEN_ID_ARRAY_TYPE`, `VLLM_INVALID_TOKEN_ID`

## Implementation Details

### Constants

```python
VLLM_TOKEN_ID_ARRAY_TYPE = "l"
```

**Purpose:**
- Type code for `array.array` to store token IDs
- `"l"` = signed long (typically 32 or 64 bits)
- Used throughout vLLM for efficient token ID storage

**Usage:**
```python
import array
token_ids = array.array(VLLM_TOKEN_ID_ARRAY_TYPE, [1, 2, 3, 4])
```

```python
VLLM_INVALID_TOKEN_ID = -1
```

**Purpose:**
- Sentinel value for invalid/unset token IDs
- Used to detect uninitialized or error states
- Distinguishes from valid token ID 0 (often padding)

### Core Data Structure: RequestMetrics

```python
@dataclass
class RequestMetrics:
    """Metrics associated with a request.

    Attributes:
        arrival_time: The time when the request arrived.
        first_scheduled_time: The time when the request was first scheduled.
        first_token_time: The time when the first token was generated.
        time_in_queue: The time the request spent in the queue.
        finished_time: The time when the request was finished.
        scheduler_time: The time spent in the scheduler when this request was
                        being considered by the scheduler.
        model_forward_time: The time spent in the model forward pass when this
                            request was in the batch.
        model_execute_time: The time spent in the model execute function. This
                            will include model forward, block/sync across
                            workers, cpu-gpu sync time and sampling time.
    """

    arrival_time: float
    last_token_time: float
    first_scheduled_time: float | None
    first_token_time: float | None
    time_in_queue: float | None
    finished_time: float | None = None
    scheduler_time: float | None = None
    model_forward_time: float | None = None
    model_execute_time: float | None = None
```

**Design Choices:**

1. **Dataclass Pattern**: Simple, immutable-like structure
2. **All Times as float**: Unix timestamps (seconds since epoch)
3. **Optional Fields**: Some metrics unavailable until certain lifecycle stages

### Timing Lifecycle

**Request Timeline:**
```
arrival_time → first_scheduled_time → first_token_time → finished_time
              ↓                      ↓
        time_in_queue          (TTFT - Time To First Token)
```

**Detailed Phases:**

1. **arrival_time** (Required)
   - When request enters the system
   - Set immediately on request creation
   - Never None

2. **last_token_time** (Required)
   - Timestamp of most recent token generation
   - Updated continuously during generation
   - Never None

3. **first_scheduled_time** (Optional until scheduled)
   - When scheduler first considers this request
   - None if not yet scheduled
   - Used to calculate queue time

4. **first_token_time** (Optional until first token)
   - When first output token is generated
   - None if still in prefill phase
   - Critical for TTFT (Time To First Token) metric

5. **time_in_queue** (Optional until scheduled)
   - Duration from arrival to first scheduling
   - Calculated: `first_scheduled_time - arrival_time`
   - None if not yet scheduled

6. **finished_time** (Optional until complete)
   - When request completes (success or failure)
   - Used to calculate total latency
   - None for in-flight requests

7. **scheduler_time** (Optional, aggregate)
   - Cumulative time spent in scheduler
   - Includes all scheduling passes for this request
   - Useful for scheduler performance analysis

8. **model_forward_time** (Optional, aggregate)
   - Cumulative time in model forward pass
   - Only the forward computation time
   - Excludes sampling, data movement

9. **model_execute_time** (Optional, aggregate)
   - Cumulative time in model execute function
   - Includes: forward + worker sync + CPU-GPU sync + sampling
   - More comprehensive than model_forward_time

### Metric Calculations

#### Time To First Token (TTFT)

```python
def calculate_ttft(metrics: RequestMetrics) -> float | None:
    if metrics.first_token_time is None:
        return None
    return metrics.first_token_time - metrics.arrival_time
```

#### Total Latency

```python
def calculate_latency(metrics: RequestMetrics) -> float | None:
    if metrics.finished_time is None:
        return None
    return metrics.finished_time - metrics.arrival_time
```

#### Inter-Token Latency (ITL)

```python
def calculate_itl(metrics: RequestMetrics, num_tokens: int) -> float | None:
    if metrics.first_token_time is None or metrics.finished_time is None:
        return None
    decode_time = metrics.finished_time - metrics.first_token_time
    decode_tokens = num_tokens - 1  # Exclude first token
    return decode_time / decode_tokens if decode_tokens > 0 else None
```

#### Throughput (tokens/second)

```python
def calculate_throughput(metrics: RequestMetrics, num_tokens: int) -> float | None:
    latency = calculate_latency(metrics)
    if latency is None or latency == 0:
        return None
    return num_tokens / latency
```

### Core Data Structure: IntermediateTensors

```python
@dataclass
class IntermediateTensors:
    """For all pipeline stages except the last, we need to return the hidden
    states and residuals to be sent to the next stage. This data structure
    contains the hidden states and residuals for a request.

    Each stage also needs to handle its own kv_connector_output.
    """

    tensors: dict[str, torch.Tensor]
    kv_connector_output: KVConnectorOutput | None

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        kv_connector_output: KVConnectorOutput | None = None,
    ) -> None:
        # manually define this function, so that
        # Dynamo knows `IntermediateTensors()` comes from this file.
        # Otherwise, dataclass will generate this function by evaluating
        # a string, and we will lose the information about the source file.
        self.tensors = tensors
        self.kv_connector_output = kv_connector_output
```

**Design Rationale:**

1. **Manual __init__**: Preserves source file information for torch.compile/Dynamo
2. **Dict of Tensors**: Flexible schema for different model architectures
3. **KV Connector Output**: Specialized data for KV cache compression/quantization

**Why Manual __init__?**
```python
# Dataclass normally generates __init__ by exec-ing a string
# This loses __code__.co_filename information
# torch.compile (Dynamo) needs correct source location for:
# - Graph caching
# - Error messages
# - Debugging
```

### IntermediateTensors: Dict-Like Interface

#### Getitem

```python
def __getitem__(self, key: str | slice):
    if isinstance(key, str):
        return self.tensors[key]
    elif isinstance(key, slice):
        return self.__class__({k: v[key] for k, v in self.tensors.items()})
```

**String Indexing:**
```python
intermediate = IntermediateTensors({
    "hidden_states": tensor1,
    "residual": tensor2
})

hidden = intermediate["hidden_states"]  # Access by name
```

**Slice Indexing:**
```python
# Slice applies to all tensors (for batch dimension slicing)
batch_subset = intermediate[0:4]
# Returns: IntermediateTensors({
#     "hidden_states": tensor1[0:4],
#     "residual": tensor2[0:4]
# })
```

#### Setitem

```python
def __setitem__(self, key: str, value: torch.Tensor):
    self.tensors[key] = value
```

**Usage:**
```python
intermediate["new_tensor"] = some_tensor
```

#### Items Iterator

```python
def __items__(self):
    return self.tensors.items()
```

**Usage:**
```python
for name, tensor in intermediate.items():
    print(f"{name}: {tensor.shape}")
```

#### Length

```python
def __len__(self):
    return len(self.tensors)
```

**Usage:**
```python
num_tensors = len(intermediate)  # Number of stored tensors
```

### Equality and Representation

```python
def __eq__(self, other: object):
    if not isinstance(other, self.__class__):
        return False
    if self.tensors.keys() != other.tensors.keys():
        return False
    return all(torch.equal(self.tensors[k], other.tensors[k]) for k in self.tensors)
```

**Equality Logic:**
1. Type check (must be same class)
2. Key check (must have same tensor names)
3. Value check (tensors must be exactly equal)

**Note:** Uses `torch.equal()` for exact equality (not approximate)

```python
def __repr__(self) -> str:
    return f"IntermediateTensors(tensors={self.tensors})"
```

**Note:** Doesn't include kv_connector_output in repr for brevity

## Usage Patterns

### RequestMetrics: Creating and Updating

```python
import time

# Create metrics on request arrival
metrics = RequestMetrics(
    arrival_time=time.time(),
    last_token_time=time.time(),
    first_scheduled_time=None,
    first_token_time=None,
    time_in_queue=None,
    finished_time=None,
    scheduler_time=None,
    model_forward_time=None,
    model_execute_time=None
)

# Update when scheduled
metrics.first_scheduled_time = time.time()
metrics.time_in_queue = metrics.first_scheduled_time - metrics.arrival_time

# Update on first token
metrics.first_token_time = time.time()
metrics.last_token_time = metrics.first_token_time

# Update on subsequent tokens
metrics.last_token_time = time.time()

# Update on completion
metrics.finished_time = time.time()
```

### RequestMetrics: Accumulating Scheduler Time

```python
# In scheduler
start = time.time()
schedule_batch()
end = time.time()

for seq in batch:
    if seq.metrics.scheduler_time is None:
        seq.metrics.scheduler_time = 0.0
    seq.metrics.scheduler_time += (end - start)
```

### RequestMetrics: Accumulating Model Time

```python
# In model executor
start = time.time()
output = model.forward(batch)
forward_end = time.time()

sample(output)
execute_end = time.time()

for seq in batch:
    # Accumulate forward time
    if seq.metrics.model_forward_time is None:
        seq.metrics.model_forward_time = 0.0
    seq.metrics.model_forward_time += (forward_end - start)

    # Accumulate execute time
    if seq.metrics.model_execute_time is None:
        seq.metrics.model_execute_time = 0.0
    seq.metrics.model_execute_time += (execute_end - start)
```

### IntermediateTensors: Pipeline Stage Communication

```python
# Stage 1 (non-final stage)
def forward_stage1(inputs):
    hidden_states = layer1(inputs)
    residual = inputs + hidden_states

    return IntermediateTensors({
        "hidden_states": hidden_states,
        "residual": residual
    })

# Stage 2 (receives intermediate tensors)
def forward_stage2(intermediate: IntermediateTensors):
    hidden_states = intermediate["hidden_states"]
    residual = intermediate["residual"]

    output = layer2(hidden_states)
    output = output + residual

    return output
```

### IntermediateTensors: Batch Slicing

```python
# Full batch intermediate tensors
full_batch = IntermediateTensors({
    "hidden": torch.randn(32, 128, 768),  # [batch, seq, hidden]
    "residual": torch.randn(32, 128, 768)
})

# Process in micro-batches
micro_batch_size = 8
for i in range(0, 32, micro_batch_size):
    micro_batch = full_batch[i:i+micro_batch_size]
    process_micro_batch(micro_batch)
```

### IntermediateTensors: With KV Connector

```python
# For models with KV cache compression
intermediate = IntermediateTensors(
    tensors={
        "hidden_states": hidden,
        "residual": residual
    },
    kv_connector_output=KVConnectorOutput(
        compressed_keys=compressed_k,
        compressed_values=compressed_v,
        compression_metadata=metadata
    )
)

# Next stage can access both
hidden = intermediate["hidden_states"]
kv_data = intermediate.kv_connector_output
```

## Integration Points

### Sequence/SequenceGroup

```python
class Sequence:
    def __init__(self, ...):
        self.metrics = RequestMetrics(
            arrival_time=time.time(),
            last_token_time=time.time(),
            ...
        )

class SequenceGroup:
    def __init__(self, ...):
        self.metrics = RequestMetrics(...)
```

### Scheduler

```python
def schedule(self):
    start = time.time()

    # Scheduling logic
    batch = select_sequences()

    end = time.time()
    elapsed = end - start

    # Update metrics for scheduled sequences
    for seq_group in batch:
        if seq_group.metrics.first_scheduled_time is None:
            seq_group.metrics.first_scheduled_time = start
            seq_group.metrics.time_in_queue = (
                start - seq_group.metrics.arrival_time
            )

        if seq_group.metrics.scheduler_time is None:
            seq_group.metrics.scheduler_time = 0.0
        seq_group.metrics.scheduler_time += elapsed
```

### Model Executor (Pipeline Parallelism)

```python
# Pipeline stage N (not last)
def execute_stage_n(inputs):
    hidden = self.model.layers[n](inputs)

    return IntermediateTensors({
        "hidden_states": hidden,
        "residual": inputs
    })

# Pipeline stage N+1
def execute_stage_n_plus_1(intermediate: IntermediateTensors):
    hidden = intermediate["hidden_states"]
    residual = intermediate["residual"]

    output = self.model.layers[n+1](hidden)
    # ...
```

### Observability/Monitoring

```python
def log_metrics(seq_group):
    metrics = seq_group.metrics

    if metrics.finished_time:
        logger.info(
            "Request completed",
            ttft=metrics.first_token_time - metrics.arrival_time,
            latency=metrics.finished_time - metrics.arrival_time,
            queue_time=metrics.time_in_queue,
            scheduler_time=metrics.scheduler_time,
            forward_time=metrics.model_forward_time,
            execute_time=metrics.model_execute_time,
        )
```

### OpenAI API Response

```python
def create_response(seq_group, outputs):
    metrics = seq_group.metrics

    return {
        "id": seq_group.request_id,
        "choices": [...],
        "usage": {...},
        # vLLM extensions
        "metrics": {
            "time_to_first_token": metrics.first_token_time - metrics.arrival_time,
            "time_in_queue": metrics.time_in_queue,
            "total_time": metrics.finished_time - metrics.arrival_time,
        }
    }
```

## Performance Considerations

### RequestMetrics Memory

Each `RequestMetrics` instance:
- 9 float fields = 72 bytes (on 64-bit Python)
- Plus object overhead ≈ 100 bytes total
- Negligible for per-request tracking

### IntermediateTensors Memory

Minimal overhead:
- Dict of tensors (one dict object)
- Tensor references (no copies)
- Total overhead: ~100 bytes per instance

**Key Point:** Tensors are not copied, only referenced

### Time Measurement Overhead

```python
# time.time() is very fast (~100ns)
start = time.time()
# ... operation ...
end = time.time()
elapsed = end - start

# Overhead: ~200ns per timing pair
# Negligible compared to model operations (milliseconds)
```

### Accumulation Pattern

```python
# Efficient accumulation (avoids repeated None checks in hot loop)
if seq.metrics.scheduler_time is None:
    seq.metrics.scheduler_time = 0.0

# Now can just add
seq.metrics.scheduler_time += elapsed
```

## Design Rationale

### Why Separate forward_time and execute_time?

**Forward Time:**
- Pure model computation
- Used to measure model throughput
- Comparable across different systems

**Execute Time:**
- Includes all overhead (sync, sampling, etc.)
- Real end-to-end time per batch
- More accurate for capacity planning

### Why Optional Times?

**Alternative:** Use sentinel values (0.0 or -1.0)
**Problem:** Ambiguous (0.0 could be actual time)

**Chosen:** Optional types (None)
**Benefits:**
- Explicit "not yet set" state
- Type checker enforces None checks
- Clear semantics

### Why Manual __init__ for IntermediateTensors?

**Problem:** Dataclass generates __init__ via exec()
```python
# Generated code loses source file info
exec(f"def __init__(self, ...): ...")
```

**Impact on torch.compile:**
- Can't determine correct source file
- Graph caching issues
- Poor error messages

**Solution:** Manual __init__ preserves __code__.co_filename

### Why Dict-Like Interface for IntermediateTensors?

**Alternative:** Named attributes (hidden_states, residual, etc.)
**Problem:** Different models need different tensors

**Chosen:** Dict-like interface
**Benefits:**
- Flexible for different architectures
- Supports arbitrary tensor names
- Can slice all tensors at once

## Testing Considerations

### RequestMetrics Tests

```python
def test_queue_time_calculation():
    metrics = RequestMetrics(
        arrival_time=0.0,
        last_token_time=0.0,
        first_scheduled_time=1.0,
        time_in_queue=None,
        ...
    )
    # Simulate queue time calculation
    metrics.time_in_queue = metrics.first_scheduled_time - metrics.arrival_time
    assert metrics.time_in_queue == 1.0

def test_ttft_calculation():
    metrics = RequestMetrics(
        arrival_time=0.0,
        last_token_time=0.0,
        first_token_time=2.5,
        ...
    )
    ttft = metrics.first_token_time - metrics.arrival_time
    assert ttft == 2.5
```

### IntermediateTensors Tests

```python
def test_tensor_access():
    tensors = {
        "hidden": torch.randn(4, 128),
        "residual": torch.randn(4, 128)
    }
    intermediate = IntermediateTensors(tensors)

    assert torch.equal(intermediate["hidden"], tensors["hidden"])
    assert len(intermediate) == 2

def test_batch_slicing():
    tensors = {
        "hidden": torch.randn(8, 128),
    }
    intermediate = IntermediateTensors(tensors)

    subset = intermediate[0:4]
    assert subset["hidden"].shape == (4, 128)
    assert isinstance(subset, IntermediateTensors)

def test_equality():
    t1 = IntermediateTensors({"a": torch.ones(2, 2)})
    t2 = IntermediateTensors({"a": torch.ones(2, 2)})
    t3 = IntermediateTensors({"a": torch.zeros(2, 2)})

    assert t1 == t2
    assert t1 != t3
```

## Related Components

- **vllm.sequence (full)**: Complete Sequence and SequenceGroup classes
- **vllm.engine.llm_engine**: Uses RequestMetrics for monitoring
- **vllm.core.scheduler**: Updates RequestMetrics during scheduling
- **vllm.worker.model_runner**: Creates IntermediateTensors for pipeline stages
- **vllm.outputs**: Includes metrics in output structures

## Future Enhancements

### RequestMetrics

1. **Per-Stage Timing**: Individual timing for each pipeline stage
2. **Memory Metrics**: Track memory usage per request
3. **Network Metrics**: For distributed inference (communication time)
4. **Sampling Metrics**: Detailed timing for sampling strategies

### IntermediateTensors

1. **Compression**: Automatic tensor compression for communication
2. **Lazy Materialization**: Defer tensor creation until needed
3. **Gradient Tracking**: Support for training/fine-tuning scenarios
4. **Memory Pinning**: Optimize for CPU-GPU transfers

## Summary

The `sequence.py` module's `RequestMetrics` and `IntermediateTensors` classes provide essential infrastructure for observability and pipeline parallelism in vLLM. `RequestMetrics` offers comprehensive timing data that enables detailed performance analysis and monitoring, while `IntermediateTensors` facilitates efficient communication between pipeline stages with minimal overhead. Together, they exemplify thoughtful design that balances performance, flexibility, and ease of use.
