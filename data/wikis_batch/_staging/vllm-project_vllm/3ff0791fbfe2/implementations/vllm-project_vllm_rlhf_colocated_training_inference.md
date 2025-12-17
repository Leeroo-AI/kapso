# RLHF Co-located Training and Inference Example

**Source:** `examples/offline_inference/rlhf_colocate.py`
**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)
**Lines:** 251

## Overview

This example demonstrates an advanced RLHF implementation where training actors and vLLM inference workers are co-located on the same GPUs. This approach maximizes GPU utilization and enables fast weight updates through CUDA Inter-Process Communication (IPC), bypassing NCCL limitations when multiple processes share a single GPU.

## Implementation Pattern

### Architecture Design

The example co-locates multiple components on the same physical GPUs:

**GPU 0 & 1:**
- Training Actor 0
- Training Actor 1
- vLLM Instance 0 (tensor parallelism across both GPUs)

**GPU 2 & 3:**
- Training Actor 2
- Training Actor 3
- vLLM Instance 1 (tensor parallelism across both GPUs)

Each component uses 0.4 GPU units, allowing 2 training actors and 1 inference instance to share the same GPU pair.

### Key Innovations

**Fractional GPU Allocation:**
Uses Ray's fractional GPU allocation (0.4 GPU per actor) to pack multiple processes on the same physical device.

**CUDA IPC Communication:**
Leverages CUDA's inter-process communication to share tensors directly in GPU memory, avoiding expensive CPU transfers and NCCL limitations.

**ZeroMQ Coordination:**
Uses ZMQ sockets to coordinate weight transfers between training actors and inference workers.

## Technical Implementation

### 1. Custom LLM Configuration

```python
class MyLLM(LLM):
    """Configure the vLLM worker for Ray placement group execution."""

    def __init__(self, *args, bundle_indices: list[int], **kwargs):
        # Prevent Ray from manipulating the top-level CUDA_VISIBLE_DEVICES variable
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # Each worker uses 0.4 GPU so that two instances fit on the same GPUs
        os.environ["VLLM_RAY_PER_WORKER_GPUS"] = "0.4"
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
        super().__init__(*args, **kwargs)
```

**Environment Variables:**
- `VLLM_RAY_PER_WORKER_GPUS="0.4"`: Allocates 40% of GPU resources to each worker
- `VLLM_RAY_BUNDLE_INDICES`: Maps workers to specific Ray placement group bundles

### 2. Training Actor Implementation

```python
class RayTrainingActor:
    """Training actor that hosts a Facebook OPT-125M model from Hugging Face."""

    def __init__(self):
        from transformers import AutoModelForCausalLM

        self.model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        self.model.to("cuda:0")

        # Zero out all the parameters for demonstration
        for name, p in self.model.named_parameters():
            p.data.zero_()
        torch.cuda.synchronize()

        # Get unique device UUID for IPC coordination
        from vllm.platforms import current_platform
        self.device_uuid = current_platform.get_device_uuid(0)
        self.zmq_context = zmq.Context()
```

**Key Features:**
- Loads model on the first visible GPU (cuda:0)
- Obtains device UUID for matching with co-located inference workers
- Initializes ZMQ context for IPC coordination

### 3. Placement Group Setup

```python
# Create placement group with 4 bundles (one per GPU)
pg = placement_group([{"GPU": 1, "CPU": 0}] * 4)
ray.get(pg.ready())

# Create training actors with fractional GPU allocation
for bundle_index in [0, 1, 2, 3]:
    training_actor = ray.remote(
        num_cpus=0,
        num_gpus=0.4,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=bundle_index,
        ),
    )(RayTrainingActor).remote()
    training_actors.append(training_actor)
```

Each training actor is assigned to a specific bundle with 0.4 GPU allocation, leaving room for inference workers.

### 4. Inference Engine Co-location

```python
for i, bundle_indices in enumerate([[0, 1], [2, 3]]):
    llm = ray.remote(
        num_cpus=0,
        num_gpus=0,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
        ),
    )(MyLLM).remote(
        model="facebook/opt-125m",
        enforce_eager=True,
        worker_extension_cls="rlhf_utils.ColocateWorkerExtension",
        tensor_parallel_size=2,
        distributed_executor_backend="ray",
        gpu_memory_utilization=0.4,
        bundle_indices=bundle_indices,
    )
    inference_engines.append(llm)
```

**Configuration:**
- `gpu_memory_utilization=0.4`: Limits memory usage to 40% of GPU capacity
- `bundle_indices=[0, 1]` or `[2, 3]`: Maps inference workers to specific GPU pairs
- `worker_extension_cls="rlhf_utils.ColocateWorkerExtension"`: Enables CUDA IPC updates

## Weight Update Protocol

### 1. ZMQ Handle Distribution

```python
# Gather all the ZMQ handles from the training actors
zmq_handles = {}
for actor in training_actors:
    zmq_handles.update(ray.get(actor.get_zmq_handles.remote()))

print(f"ZMQ handles: {zmq_handles}")
```

Each training actor creates a unique ZMQ socket address (IPC Unix socket) identified by its device UUID.

### 2. Tensor Flattening and IPC Transfer

**Training Actor Side:**
```python
def update_weights(self):
    align_size = 256

    def get_size(p: torch.Tensor) -> int:
        return (p.nbytes + align_size - 1) // align_size * align_size

    named_parameters = dict(self.model.named_parameters())
    max_tensor_size = max(get_size(p) for p in named_parameters.values())

    # Create shared buffer for IPC
    buffer = torch.empty(max_tensor_size * 2, dtype=torch.uint8, device="cuda:0")

    # Bind ZMQ socket and send IPC handle
    s = self.zmq_context.socket(zmq.REQ)
    s.bind(self.zmq_handle)
    handle = reduce_tensor(buffer)

    # Bucket parameters for batch transfer
    buckets = []
    for name, p in named_parameters.items():
        # Copy parameter data to buffer
        buffer[offset : offset + p.nbytes].data.copy_(
            p.data.view(-1).view(dtype=torch.uint8), non_blocking=True
        )
        torch.cuda.synchronize()

    # Send metadata over ZMQ
    s.send_pyobj(named_tensors)
```

**Worker Extension Side:**
```python
def update_weights_from_ipc(self, zmq_handles: dict[str, str]):
    socket = self._zmq_ctx.socket(zmq.REP)
    socket.connect(zmq_handles[self.report_device_id()])

    while True:
        payload = socket.recv_pyobj()

        if payload is None:
            # Update complete
            break

        if isinstance(payload, tuple):
            # Rebuild IPC buffer handle
            buffer = rebuild_ipc(payload, self.device.index)
            socket.send(b"")
            continue

        # Extract tensors from buffer
        for item in payload:
            tensor = buffer[offset : offset + size].view(dtype=dtype).view(shape)
            weights.append((item["name"], tensor))

        self.model_runner.model.load_weights(weights=weights)
```

### 3. Placement Verification

```python
# Verify placement: the first two training actors share the same GPUs as
# the first inference engine
assert training_actor_device_ids[:2] == inference_engine_device_ids[0]

# Verify placement: the last two training actors share the same GPUs as
# the second inference engine
assert training_actor_device_ids[2:] == inference_engine_device_ids[1]
```

The example validates that training actors and inference workers are correctly co-located by comparing device UUIDs.

## CUDA IPC Advantages

### Why CUDA IPC?

**NCCL Limitations:**
- NCCL typically assumes one process per GPU
- Multiple processes on the same GPU can cause collectives to hang
- Not designed for fractional GPU sharing scenarios

**CUDA IPC Benefits:**
- Direct GPU memory sharing between processes
- No CPU involvement in data transfer
- Supports multiple processes on the same GPU
- Lower latency for co-located processes

### IPC Handle Reconstruction

```python
def rebuild_ipc(
    handle: tuple[Callable, tuple], device_id: int | None = None
) -> torch.Tensor:
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        # Change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
    buffer = func(*list_args)
    return buffer
```

This helper adjusts the device ID in the IPC handle to account for different CUDA_VISIBLE_DEVICES mappings between processes.

## Memory Management

### Alignment and Bucketing

```python
align_size = 256

def get_size(p: torch.Tensor) -> int:
    return (p.nbytes + align_size - 1) // align_size * align_size
```

Parameters are aligned to 256-byte boundaries to avoid misaligned memory access, which can cause GPU errors.

### Buffer Cleanup

```python
del buffer
gc.collect()
torch.cuda.empty_cache()
```

Explicit cleanup ensures GPU memory is released promptly after weight updates, preventing memory leaks in long-running RLHF training loops.

## Usage Requirements

### System Requirements
- Single-node cluster with 4 GPUs (multi-node supported by Ray)
- Exclusive GPU access during vLLM initialization
- CUDA IPC support (most modern NVIDIA GPUs)

### Dependencies
- Ray for distributed execution and placement groups
- PyTorch for model operations
- ZeroMQ (zmq) for IPC coordination
- Transformers for model loading
- vLLM with worker extension support

### Configuration Notes

**GPU Memory Utilization:**
With 0.4 allocation per component, ensure total usage doesn't exceed GPU capacity. The example uses:
- 2 training actors × 0.4 = 0.8 GPU
- 1 inference worker × ~0.4 GPU (memory utilization setting)

**Environment Variables:**
Critical environment variables must be set before LLM initialization:
- `VLLM_RAY_PER_WORKER_GPUS`: Controls GPU fraction per worker
- `VLLM_RAY_BUNDLE_INDICES`: Maps workers to placement bundles

## Performance Characteristics

### Advantages
- **Maximum GPU Utilization:** Multiple workloads on the same GPU
- **Fast Weight Updates:** Direct GPU memory sharing via IPC
- **Reduced Latency:** No CPU bottleneck in weight transfer
- **Resource Efficiency:** Fewer total GPUs required

### Trade-offs
- **Complex Setup:** More intricate than separated GPU approach
- **Careful Memory Management:** Must manually track allocations
- **Debugging Difficulty:** IPC issues can be harder to diagnose

## Production Considerations

### When to Use Co-location

**Ideal Scenarios:**
- Limited GPU availability
- Tight coupling between training and inference
- High-frequency weight updates
- Cost optimization is critical

**Alternative Approach:**
Use the separated GPU pattern (rlhf.py) when:
- Abundant GPU resources available
- Independent scaling of training/inference
- Simpler operational requirements

### Scalability

The pattern scales by:
- Increasing placement group size (more GPU bundles)
- Creating more training actor / inference engine pairs
- Extending to multi-node Ray clusters

## Related Examples

- `rlhf.py`: Separated training and inference on different GPUs
- `rlhf_online_quant.py`: RLHF with quantization
- `rlhf_utils.py`: Contains ColocateWorkerExtension implementation

## References

- **Ray Placement Groups:** [Documentation](https://docs.ray.io/en/latest/placement-groups.html)
- **CUDA IPC:** PyTorch multiprocessing reductions
- **ZeroMQ:** High-performance asynchronous messaging library
