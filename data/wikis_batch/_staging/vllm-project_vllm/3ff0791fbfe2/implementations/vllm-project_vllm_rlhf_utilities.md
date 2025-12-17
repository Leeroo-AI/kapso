# RLHF Utilities - Worker Extensions

**Source:** `examples/offline_inference/rlhf_utils.py`
**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)
**Lines:** 168

## Overview

This module provides utility classes and functions for implementing RLHF workflows with vLLM. It contains two worker extension classes that enable different weight synchronization patterns: NCCL-based broadcast for separated processes and CUDA IPC for co-located processes.

## Core Components

### 1. Stateless Process Group Initialization

```python
def stateless_init_process_group(master_address, master_port, rank, world_size, device):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl
```

**Purpose:**
Creates a process group for NCCL communication without interfering with PyTorch's global `torch.distributed` state. This is essential for vLLM's architecture which manages its own distributed communication.

**Parameters:**
- `master_address`: IP address of the coordinator process
- `master_port`: Port for process group coordination
- `rank`: Process rank in the group (0-based)
- `world_size`: Total number of processes in the group
- `device`: CUDA device for NCCL operations

**Returns:**
A `PyNcclCommunicator` instance that provides NCCL collective operations.

### 2. WorkerExtension Class

```python
class WorkerExtension:
    """
    The class for vLLM's worker to inherit from.
    By defining an extension class, the code can work no matter what is
    the underlying worker class.

    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.
    """
```

This extension enables NCCL-based weight updates for separated training/inference scenarios (used in `rlhf.py` and `rlhf_online_quant.py`).

#### Method: init_weight_update_group

```python
def init_weight_update_group(
    self, master_address, master_port, rank_offset, world_size
):
    from vllm.distributed.parallel_state import get_world_group

    rank = get_world_group().rank + rank_offset
    self.model_update_group = stateless_init_process_group(
        master_address,
        master_port,
        rank,
        world_size,
        self.device,
    )
```

**Purpose:**
Initializes the NCCL communication group for receiving weight updates from the training process.

**Key Details:**
- Computes rank by adding `rank_offset` to worker's tensor-parallel rank
- Creates a stateless process group that won't conflict with vLLM's internal groups
- Stores the communicator in `self.model_update_group` for later use

**Rank Calculation Example:**
For tensor parallelism size 2 with training on rank 0:
- Worker 0: rank = 0 + 1 = 1
- Worker 1: rank = 1 + 1 = 2
- Training process: rank = 0

#### Method: update_weight

```python
def update_weight(self, name, dtype_name, shape):
    dtype = getattr(torch, dtype_name)
    weight = torch.empty(shape, dtype=dtype, device="cuda")
    self.model_update_group.broadcast(
        weight, src=0, stream=torch.cuda.current_stream()
    )

    self.model_runner.model.load_weights(weights=[(name, weight)])

    del weight
```

**Purpose:**
Receives a single weight tensor from the training process via NCCL broadcast and loads it into the model.

**Process:**
1. Creates an empty tensor with the correct shape and dtype
2. Receives data via NCCL broadcast from source rank 0 (training process)
3. Loads the weight into the model using vLLM's weight loading mechanism
4. Explicitly deletes the temporary tensor to free memory

**Memory Efficiency:**
The temporary tensor is deleted immediately after loading to minimize peak memory usage during weight updates.

#### Method: check_weights_changed

```python
def check_weights_changed(self):
    """
    Check if the weights are updated to 0.
    """
    weights_updated = True
    for name, p in self.model_runner.model.named_parameters():
        weights_updated = weights_updated and torch.allclose(p, torch.zeros_like(p))
    return weights_updated
```

**Purpose:**
Verifies that weight updates were successfully applied. In the example workflows, weights are zeroed out for demonstration, so this checks all parameters are zero.

**Production Usage:**
In real applications, replace this with checksums or version numbers to verify synchronization.

## CUDA IPC Components

### 3. IPC Buffer Reconstruction

```python
def rebuild_ipc(
    handle: tuple[Callable, tuple], device_id: int | None = None
) -> torch.Tensor:
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
    buffer = func(*list_args)
    return buffer
```

**Purpose:**
Reconstructs a CUDA tensor from an IPC handle, adjusting for different device IDs between processes.

**Background:**
PyTorch's `reduce_tensor()` function creates an IPC handle (tuple of function and args) that can be passed to another process. The receiving process calls `rebuild_ipc()` to reconstruct the tensor.

**Device ID Adjustment:**
Critical for co-location scenarios where processes have different `CUDA_VISIBLE_DEVICES` mappings:
- Process A sees GPUs [0, 1, 2, 3]
- Process B sees GPUs [2, 3] (mapped to local [0, 1])
- IPC handle must translate device IDs appropriately

### 4. FlattenedTensorMetadata

```python
class FlattenedTensorMetadata(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    # specify the start offset of this tensor in shared ipc_buffer tensor
    offset: int
```

**Purpose:**
Describes a tensor's location within a flattened IPC buffer.

**Usage Pattern:**
Multiple tensors are packed into a single large buffer to amortize IPC overhead. Metadata describes how to extract individual tensors:

```python
# Extract tensor from buffer
size = dtype.itemsize * shape.numel()
tensor = buffer[offset : offset + size].view(dtype=dtype).view(shape)
```

### 5. ColocateWorkerExtension Class

```python
class ColocateWorkerExtension:
    """
    The class for vLLM's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class.

    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.
    """
```

This extension enables CUDA IPC-based weight updates for co-located training/inference scenarios (used in `rlhf_colocate.py`).

#### Method: update_weights_from_ipc

```python
def update_weights_from_ipc(self, zmq_handles: dict[str, str]):
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    assert self.device is not None
    if not hasattr(self, "_zmq_ctx") or self._zmq_ctx is None:
        self._zmq_ctx = zmq.Context()

    socket = self._zmq_ctx.socket(zmq.REP)
    socket.connect(zmq_handles[self.report_device_id()])

    buffer: torch.Tensor | None = None

    while True:
        payload = socket.recv_pyobj()

        if payload is None:
            # Update complete
            process_weights_after_loading(
                self.model_runner.model, self.model_config, self.device
            )
            torch.cuda.synchronize()
            socket.send(b"")
            break

        if isinstance(payload, tuple):
            # IPC handle for shared buffer
            buffer = rebuild_ipc(payload, self.device.index)
            assert buffer.dtype == torch.uint8
            socket.send(b"")
            continue

        # Payload is list of FlattenedTensorMetadata
        assert isinstance(payload, list)
        assert buffer is not None

        weights = []
        for item in payload:
            shape = item["shape"]
            if isinstance(shape, (list, tuple)):
                shape = torch.Size(shape)

            dtype, offset = item["dtype"], item["offset"]
            size = dtype.itemsize * shape.numel()
            tensor = buffer[offset : offset + size].view(dtype=dtype).view(shape)
            weights.append((item["name"], tensor))

        self.model_runner.model.load_weights(weights=weights)
        del weights
        torch.cuda.synchronize()
        socket.send(b"")

    socket.close()
    del buffer
    gc.collect()
    torch.cuda.empty_cache()
```

**Protocol Flow:**

1. **Socket Connection:**
   - Connects to ZMQ socket identified by device UUID
   - Uses REQ-REP pattern for synchronous communication

2. **Buffer Reception:**
   - Receives IPC handle as tuple
   - Reconstructs shared GPU buffer using `rebuild_ipc()`
   - Buffer contains flattened parameter data

3. **Batch Weight Updates:**
   - Receives lists of `FlattenedTensorMetadata`
   - Extracts tensors from buffer using offset/size information
   - Loads batch of weights into model

4. **Completion:**
   - `None` payload signals end of update
   - Calls `process_weights_after_loading()` for post-processing
   - Synchronizes GPU and cleans up resources

**Memory Management:**
Explicit cleanup with `gc.collect()` and `torch.cuda.empty_cache()` ensures GPU memory is released, crucial for tight memory scenarios with fractional GPU allocation.

#### Method: report_device_id

```python
def report_device_id(self) -> str:
    from vllm.platforms import current_platform

    self.device_uuid = current_platform.get_device_uuid(self.device.index)
    return self.device_uuid
```

**Purpose:**
Returns the GPU's unique device UUID for matching co-located processes.

**Device UUID vs. Device Index:**
- Device index (e.g., 0, 1, 2) depends on `CUDA_VISIBLE_DEVICES`
- Device UUID is globally unique and independent of visibility settings
- Essential for matching processes that see different device indices

#### Method: check_weights_changed

Identical to `WorkerExtension.check_weights_changed()` - verifies weights were updated to zero for demonstration purposes.

## Usage Patterns

### Pattern 1: NCCL-Based Updates (Separated GPUs)

```python
# In main script
llm = LLM(
    model="...",
    worker_extension_cls="rlhf_utils.WorkerExtension",
    ...
)

# Initialize communication
handle = llm.collective_rpc.remote(
    "init_weight_update_group",
    args=(master_address, master_port, rank_offset, world_size)
)
ray.get(handle)

# Update weights
for name, param in model.named_parameters():
    handle = llm.collective_rpc.remote(
        "update_weight",
        args=(name, dtype_name, shape)
    )
    nccl_group.broadcast(param, src=0)
    ray.get(handle)
```

### Pattern 2: CUDA IPC Updates (Co-located GPUs)

```python
# In main script
llm = LLM(
    model="...",
    worker_extension_cls="rlhf_utils.ColocateWorkerExtension",
    ...
)

# Gather ZMQ handles
zmq_handles = {}
for actor in training_actors:
    zmq_handles.update(ray.get(actor.get_zmq_handles.remote()))

# Trigger updates in parallel
ray.get([
    actor.update_weights.remote() for actor in training_actors
] + [
    llm.collective_rpc.remote("update_weights_from_ipc", args=(zmq_handles,))
    for llm in inference_engines
])
```

## Design Principles

### Extension Pattern

**Why Separate Module:**
```python
# Main script passes fully qualified name
worker_extension_cls="rlhf_utils.WorkerExtension"
```

This allows vLLM to:
- Import the extension class dynamically
- Work with any underlying worker implementation
- Avoid circular dependencies

**Inheritance Mechanism:**
vLLM's worker classes dynamically inherit from the extension, adding the custom methods to the worker's API.

### Stateless Communication

The `StatelessProcessGroup` design enables:
- Multiple independent process groups in the same application
- No global state conflicts with PyTorch distributed
- Clean separation between vLLM internal and external communication

### Memory Safety

Both extensions emphasize explicit memory management:
- Delete temporary tensors immediately after use
- Call `gc.collect()` and `torch.cuda.empty_cache()`
- Synchronize CUDA streams before cleanup

## Production Considerations

### Error Handling

The examples omit error handling for clarity. Production code should add:

```python
def update_weight(self, name, dtype_name, shape):
    try:
        dtype = getattr(torch, dtype_name)
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        self.model_update_group.broadcast(weight, src=0)
        self.model_runner.model.load_weights(weights=[(name, weight)])
    except Exception as e:
        logger.error(f"Failed to update weight {name}: {e}")
        raise
    finally:
        if 'weight' in locals():
            del weight
```

### Weight Validation

Replace the simple zero-check with robust validation:

```python
def check_weights_changed(self, expected_checksum):
    """Verify weights match expected checksum."""
    import hashlib

    checksum = hashlib.sha256()
    for name, p in self.model_runner.model.named_parameters():
        checksum.update(p.cpu().numpy().tobytes())

    return checksum.hexdigest() == expected_checksum
```

### Communication Timeouts

Add timeouts to prevent hanging on failed updates:

```python
# ZMQ socket timeout
socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30 seconds

# NCCL timeout via environment variable
os.environ["NCCL_TIMEOUT"] = "300"  # 5 minutes
```

## Related Examples

- **rlhf.py:** Uses `WorkerExtension` for NCCL-based weight updates
- **rlhf_colocate.py:** Uses `ColocateWorkerExtension` for IPC-based updates
- **rlhf_online_quant.py:** Uses `WorkerExtension` with quantized models

## References

- **StatelessProcessGroup:** vLLM's distributed utilities
- **PyNcclCommunicator:** vLLM's NCCL wrapper
- **CUDA IPC:** PyTorch multiprocessing tensor reduction
- **ZeroMQ:** High-performance asynchronous messaging
