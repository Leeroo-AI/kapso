# RLHF Training-Inference Separation Example

**Source:** `examples/offline_inference/rlhf.py`
**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)
**Lines:** 147

## Overview

This example demonstrates how to implement Reinforcement Learning from Human Feedback (RLHF) using vLLM and Ray with separated training and inference workloads. The implementation showcases a production-ready pattern for distributing training and inference across different GPUs with efficient weight synchronization.

## Implementation Pattern

### Architecture Design

The example uses a clear separation of concerns:

**Training Process (GPU 0):**
- Hosts a Hugging Face Transformer model (facebook/opt-125m)
- Performs weight updates based on RL objectives
- Acts as the source of truth for model weights

**Inference Engine (GPUs 1-2):**
- Runs vLLM with tensor parallelism across 2 GPUs
- Serves inference requests with low latency
- Receives weight updates from the training process

### Key Components

**MyLLM Class:**
```python
class MyLLM(LLM):
    """Configure the vLLM worker for Ray placement group execution."""

    def __init__(self, *args, **kwargs):
        # Remove the top-level CUDA_VISIBLE_DEVICES variable set by Ray
        # so that vLLM can manage its own device placement within the worker.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        super().__init__(*args, **kwargs)
```

This custom LLM class ensures proper device placement within Ray's execution environment by removing Ray's CUDA_VISIBLE_DEVICES setting, allowing vLLM to handle its own GPU allocation.

## Technical Implementation

### 1. Resource Allocation

```python
# Training model on GPU 0
train_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
train_model.to("cuda:0")

# Configure Ray to use GPUs 1-2 for inference
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
ray.init()

# Create placement group for inference engine
pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * 2)
ray.get(pg_inference.ready())
```

The implementation uses Ray placement groups to explicitly control GPU allocation, ensuring training and inference processes don't compete for resources.

### 2. Inference Engine Setup

```python
llm = ray.remote(
    num_cpus=0,
    num_gpus=0,
    scheduling_strategy=scheduling_inference,
)(MyLLM).remote(
    model="facebook/opt-125m",
    enforce_eager=True,
    worker_extension_cls="rlhf_utils.WorkerExtension",
    tensor_parallel_size=2,
    distributed_executor_backend="ray",
)
```

**Configuration Details:**
- `enforce_eager=True`: Reduces startup latency by avoiding graph compilation
- `worker_extension_cls`: Enables custom weight update logic
- `tensor_parallel_size=2`: Splits the model across 2 GPUs
- `distributed_executor_backend="ray"`: Uses Ray for distributed execution

### 3. Weight Synchronization

**Establishing Communication Channel:**
```python
master_address = get_ip()
master_port = get_open_port()

handle = llm.collective_rpc.remote(
    "init_weight_update_group", args=(master_address, master_port, 1, 3)
)

model_update_group = stateless_init_process_group(
    master_address, master_port, 0, 3, torch.device("cuda:0")
)
ray.get(handle)
```

Creates a stateless process group for NCCL communication between training and inference processes.

**Broadcasting Updates:**
```python
for name, p in train_model.named_parameters():
    dtype_name = str(p.dtype).split(".")[-1]
    handle = llm.collective_rpc.remote(
        "update_weight", args=(name, dtype_name, p.shape)
    )
    model_update_group.broadcast(p, src=0, stream=torch.cuda.current_stream())
    ray.get(handle)
```

Uses NCCL broadcast for efficient GPU-to-GPU weight transfer without involving CPU memory.

## Workflow Demonstration

### Step 1: Initial Inference
```python
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0)
outputs = ray.get(llm.generate.remote(prompts, sampling_params))
```

Generates baseline outputs using the original model weights.

### Step 2: Simulated Training Step
```python
# Simulate a training step by zeroing out all model weights
for name, p in train_model.named_parameters():
    p.data.zero_()
```

In production, this would apply gradients from an RL objective (e.g., PPO on a reward model).

### Step 3: Weight Synchronization
The updated weights are broadcast to all inference workers, ensuring consistency across the distributed system.

### Step 4: Verification
```python
assert all(ray.get(llm.collective_rpc.remote("check_weights_changed")))
```

Confirms that all inference workers received the weight updates.

## Usage Requirements

### System Requirements
- Single-node cluster with at least 3 GPUs
- GPUs exclusively dedicated to vLLM (no concurrent workloads)
- Sufficient GPU memory for both training and inference models

### Dependencies
- Ray for distributed execution
- PyTorch for model operations
- Transformers for model loading
- vLLM for inference serving

### Configuration Considerations

**GPU Visibility:**
The example carefully manages GPU visibility to ensure proper resource isolation between training and inference components.

**Memory Profiling:**
vLLM performs memory profiling during initialization. Residual GPU activity from other processes can interfere with this profiling and cause unexpected behavior.

## Production Considerations

### Reference Implementation
The example notes that for production deployments supporting multiple training and inference replicas, the [OpenRLHF framework](https://github.com/OpenRLHF/OpenRLHF) provides a more complete solution.

### Scalability
While this example uses a single-node setup, Ray supports multi-node clusters, enabling:
- Multiple training actors
- Multiple inference replicas
- Horizontal scaling of both components

### Performance Optimization
- **Tensor Parallelism:** Enables handling larger models by splitting across GPUs
- **Eager Execution:** Reduces startup time for faster iteration during development
- **NCCL Communication:** Provides high-bandwidth GPU-to-GPU transfers

## Integration Points

### Worker Extension
The example relies on `rlhf_utils.WorkerExtension` which provides:
- `init_weight_update_group`: Sets up the communication channel
- `update_weight`: Receives and applies weight updates
- `check_weights_changed`: Validates synchronization

### Ray Collective RPC
Uses Ray's collective RPC mechanism to invoke methods across distributed workers, enabling coordinated operations across the inference cluster.

## Use Cases

This implementation pattern is valuable for:

1. **RLHF Training Pipelines:** Separating policy optimization from rollout generation
2. **Online Learning Systems:** Updating serving models without downtime
3. **A/B Testing:** Running multiple model versions with different weights
4. **Continuous Training:** Incrementally updating production models

## Related Examples

- `rlhf_colocate.py`: Co-located training and inference on same GPUs
- `rlhf_online_quant.py`: RLHF with quantization for memory efficiency
- `rlhf_utils.py`: Shared utilities for RLHF examples

## References

- **vLLM Distributed Execution:** Uses Ray as the distributed executor backend
- **StatelessProcessGroup:** Enables NCCL communication without torch.distributed global state
- **Ray Placement Groups:** [Ray Documentation](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html)
