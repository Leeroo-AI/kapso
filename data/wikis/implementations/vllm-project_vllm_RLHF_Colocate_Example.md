{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::RLHF]], [[domain::Memory Optimization]], [[domain::CUDA IPC]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
This example demonstrates memory-efficient RLHF by co-locating training and inference on the same GPUs using CUDA IPC.

=== Description ===
The RLHF colocate example shows an advanced pattern for memory-efficient RLHF training where vLLM inference workers and training actors share the same physical GPUs. Unlike the standard RLHF example which uses separate GPUs, this approach uses Ray placement groups to allocate fractional GPU resources (0.4 GPU per process) and CUDA Inter-Process Communication (IPC) for fast weight synchronization without NCCL. The example demonstrates handling 4 GPUs with 2 training actors and 1 vLLM instance per GPU pair, maximizing hardware utilization for resource-constrained environments.

=== Usage ===
Use this pattern when GPU memory is limited and you need to share GPUs between training and inference, want to minimize weight transfer overhead using CUDA IPC, need maximum GPU utilization efficiency, or are building cost-optimized RLHF systems. Requires careful memory profiling to avoid OOM errors.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/rlhf_colocate.py examples/offline_inference/rlhf_colocate.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
python examples/offline_inference/rlhf_colocate.py
</syntaxhighlight>

Note: Requires 4 GPUs. Example creates 4 training actors + 2 vLLM instances.

== Key Concepts ==

=== Fractional GPU Allocation ===
Ray placement groups enable GPU sharing:
* Each bundle represents 1 physical GPU
* Training actors use 0.4 GPU each
* vLLM instances use remaining GPU capacity
* '''VLLM_RAY_PER_WORKER_GPUS=0.4''': Configure vLLM worker GPU fraction
* '''VLLM_RAY_BUNDLE_INDICES''': Map workers to placement bundles

=== CUDA IPC Weight Transfer ===
CUDA IPC enables direct GPU memory sharing:
* Training actors expose tensors via CUDA IPC handles
* vLLM workers map tensors directly in GPU memory
* Bypasses NCCL limitations for processes sharing GPUs
* Uses ZMQ for handle exchange and synchronization
* Significantly faster than CPU round-tripping

=== Colocation Architecture ===
The example demonstrates specific GPU mapping:
* '''GPUs 0-1''': Training actors 0-1 + vLLM instance 0 (TP=2)
* '''GPUs 2-3''': Training actors 2-3 + vLLM instance 1 (TP=2)
* Each vLLM instance uses 2-way tensor parallelism
* Training actors verify device placement before weight updates

=== Custom Worker Extension ===
ColocateWorkerExtension (from rlhf_utils.py) provides:
* '''update_weights_from_ipc()''': Receive weights via CUDA IPC
* '''report_device_id()''': Return GPU UUID for mapping
* '''check_weights_changed()''': Verify weight synchronization
* Handles tensor alignment and memory cleanup

== Usage Examples ==

=== Custom MyLLM for Colocation ===
<syntaxhighlight lang="python">
import os
from vllm import LLM

class MyLLM(LLM):
    """Configure vLLM worker for Ray placement group with GPU sharing."""

    def __init__(self, *args, bundle_indices: list[int], **kwargs):
        # Prevent Ray from overriding CUDA_VISIBLE_DEVICES
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        # Each worker uses 0.4 GPU (2 workers per GPU)
        os.environ["VLLM_RAY_PER_WORKER_GPUS"] = "0.4"

        # Specify which placement bundles this instance uses
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))

        print(f"Creating LLM with bundle_indices={bundle_indices}")
        super().__init__(*args, **kwargs)
</syntaxhighlight>

=== Training Actor with IPC Support ===
<syntaxhighlight lang="python">
import torch
import zmq
from transformers import AutoModelForCausalLM
from vllm.platforms import current_platform

class RayTrainingActor:
    """Training actor that shares weights via CUDA IPC."""

    def __init__(self):
        # Load model on first visible GPU
        self.model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        self.model.to("cuda:0")

        # Zero out parameters (simulate training)
        for name, p in self.model.named_parameters():
            p.data.zero_()
        torch.cuda.synchronize()

        # Get GPU UUID for IPC mapping
        self.device_uuid = current_platform.get_device_uuid(0)

        # Setup ZMQ for handle exchange
        self.zmq_context = zmq.Context()
        self.zmq_address_counter = 0
        self.zmq_handle = None

    def report_device_id(self) -> str:
        return self.device_uuid

    def get_zmq_handles(self) -> dict[str, str]:
        """Create ZMQ handle for this actor's GPU."""
        suffix = f"{self.device_uuid}-{self.zmq_address_counter}"
        self.zmq_handle = f"ipc:///tmp/rl-colocate-zmq-{suffix}.sock"
        self.zmq_address_counter += 1
        return {self.device_uuid: self.zmq_handle}
</syntaxhighlight>

=== Placement Group Setup ===
<syntaxhighlight lang="python">
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
ray.init()

# Create placement group: 1 bundle per GPU
pg = placement_group([{"GPU": 1, "CPU": 0}] * 4)
ray.get(pg.ready())

# Create training actors (4 actors, one per GPU)
training_actors = []
for bundle_index in [0, 1, 2, 3]:
    actor = ray.remote(
        num_cpus=0,
        num_gpus=0.4,  # Fractional GPU
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=bundle_index,
        ),
    )(RayTrainingActor).remote()
    training_actors.append(actor)

# Create vLLM instances (2 instances, TP=2 each)
inference_engines = []
for bundle_indices in [[0, 1], [2, 3]]:
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
</syntaxhighlight>

=== Verify Device Placement ===
<syntaxhighlight lang="python">
# Get device IDs from training actors
training_actor_device_ids = []
for i, actor in enumerate(training_actors):
    device_id = ray.get(actor.report_device_id.remote())
    print(f"Training actor {i} on {device_id}")
    training_actor_device_ids.append(device_id)

# Get device IDs from inference engines
inference_engine_device_ids = []
for i, llm in enumerate(inference_engines):
    device_ids = ray.get(llm.collective_rpc.remote("report_device_id", args=tuple()))
    print(f"Inference engine {i} on {device_ids}")
    inference_engine_device_ids.append(device_ids)

# Verify colocation
assert training_actor_device_ids[:2] == inference_engine_device_ids[0]
assert training_actor_device_ids[2:] == inference_engine_device_ids[1]
print("Colocation verified!")
</syntaxhighlight>

=== Weight Synchronization via IPC ===
<syntaxhighlight lang="python">
# Gather ZMQ handles from all training actors
print("Gathering ZMQ handles from training actors...")
zmq_handles = {}
for actor in training_actors:
    zmq_handles.update(ray.get(actor.get_zmq_handles.remote()))

print(f"ZMQ handles: {zmq_handles}")

# Update weights via CUDA IPC
print("Updating weights via CUDA IPC...")
ray.get(
    # Training actors prepare weights for IPC
    [actor.update_weights.remote() for actor in training_actors]
    +
    # vLLM workers consume weights via IPC
    [
        llm.collective_rpc.remote("update_weights_from_ipc", args=(zmq_handles,))
        for llm in inference_engines
    ]
)

# Verify weights updated
print("Verifying weight updates...")
for llm in inference_engines:
    assert ray.get(llm.collective_rpc.remote("check_weights_changed", args=tuple()))

print("Weight synchronization complete!")
</syntaxhighlight>

=== Memory Profiling Considerations ===
<syntaxhighlight lang="python">
# Monitor GPU memory during colocation
import torch

def print_gpu_memory():
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# Check before vLLM initialization
print("Memory before vLLM:")
print_gpu_memory()

# vLLM memory profiling runs here

# Check after initialization
print("Memory after vLLM:")
print_gpu_memory()

# Adjust gpu_memory_utilization if needed
# Lower values leave more room for training actors
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[related_to::Implementation:vllm-project_vllm_RLHF_Example]]
* [[related_to::Implementation:vllm-project_vllm_RLHF_Utils]]
