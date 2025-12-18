{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::RLHF]], [[domain::Worker Extensions]], [[domain::CUDA IPC]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
This utility module provides worker extensions and helper functions for integrating vLLM with RLHF training workflows.

=== Description ===
The RLHF utils module contains reusable components for building RLHF systems with vLLM. It provides two worker extension classes (WorkerExtension for standard RLHF and ColocateWorkerExtension for memory-efficient colocation) and utility functions for process group initialization and CUDA IPC handling. These components abstract the complexity of weight synchronization, GPU placement, and inter-process communication, making it easier to build custom RLHF training pipelines. The module demonstrates how to extend vLLM workers with custom functionality without modifying core vLLM code.

=== Usage ===
Import and use these utilities when building RLHF systems, implementing custom worker extensions, handling weight synchronization between training and inference, or managing CUDA IPC for colocated processes. Reference these classes via worker_extension_cls parameter when creating LLM instances.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/rlhf_utils.py examples/offline_inference/rlhf_utils.py]

=== Module Usage ===
<syntaxhighlight lang="python">
# Import in RLHF examples
from rlhf_utils import stateless_init_process_group

# Reference extension class by fully qualified name
llm = LLM(
    model="facebook/opt-125m",
    worker_extension_cls="rlhf_utils.WorkerExtension",
    # or
    worker_extension_cls="rlhf_utils.ColocateWorkerExtension",
)
</syntaxhighlight>

== Key Concepts ==

=== Worker Extension Pattern ===
Worker extensions augment vLLM workers with custom methods:
* Inherit from base worker class automatically
* Methods callable via collective_rpc
* Access to worker internals (model, device, etc.)
* Fully qualified class name passed as string

=== Stateless Process Groups ===
StatelessProcessGroup enables external process coordination:
* No dependency on torch.distributed global state
* Creates independent NCCL communicators
* Supports mixed training/inference process groups
* Ideal for multi-framework scenarios (PyTorch + vLLM)

=== CUDA IPC Tensor Reconstruction ===
CUDA IPC enables direct GPU memory mapping:
* Tensors shared between processes on same GPU
* Avoids CPU round-trips and copying
* Requires device ID remapping for CUDA_VISIBLE_DEVICES
* Used in colocation scenarios

== Component Overview ==

=== stateless_init_process_group() ===
Creates a NCCL process group independent of torch.distributed:
* '''Parameters''': master_address, master_port, rank, world_size, device
* '''Returns''': PyNcclCommunicator for collective operations
* '''Use case''': Connecting external training processes with vLLM workers

=== WorkerExtension ===
Standard RLHF worker extension for separate training/inference:
* '''init_weight_update_group()''': Initialize NCCL communication
* '''update_weight()''': Receive weight updates via broadcast
* '''check_weights_changed()''': Verify synchronization

=== ColocateWorkerExtension ===
Colocation-optimized worker extension using CUDA IPC:
* '''update_weights_from_ipc()''': Receive weights via CUDA IPC
* '''report_device_id()''': Return GPU UUID for mapping
* '''check_weights_changed()''': Verify synchronization

=== rebuild_ipc() ===
Reconstructs tensor from CUDA IPC handle:
* Handles device ID remapping for different CUDA_VISIBLE_DEVICES
* Required when processes have different GPU visibility
* Returns mapped tensor in local GPU memory

== Usage Examples ==

=== Creating Stateless Process Group ===
<syntaxhighlight lang="python">
import torch
from rlhf_utils import stateless_init_process_group
from vllm.utils.network_utils import get_ip, get_open_port

# Setup communication parameters
master_address = get_ip()
master_port = get_open_port()

# Training process (rank 0)
train_pg = stateless_init_process_group(
    master_address=master_address,
    master_port=master_port,
    rank=0,
    world_size=3,  # 1 training + 2 inference workers
    device=torch.device("cuda:0")
)

# Use for collective operations
for name, param in model.named_parameters():
    train_pg.broadcast(param, src=0, stream=torch.cuda.current_stream())
</syntaxhighlight>

=== Using WorkerExtension ===
<syntaxhighlight lang="python">
import ray
from vllm import LLM

# Create LLM with worker extension
llm = ray.remote(num_gpus=2)(LLM).remote(
    model="facebook/opt-125m",
    worker_extension_cls="rlhf_utils.WorkerExtension",
    tensor_parallel_size=2,
)

# Initialize weight update group (called on workers)
handle = llm.collective_rpc.remote(
    "init_weight_update_group",
    args=(master_address, master_port, 1, 3)
)
ray.get(handle)

# Update specific weight
handle = llm.collective_rpc.remote(
    "update_weight",
    args=("model.layers.0.weight", "float32", (4096, 4096))
)
# Training process broadcasts weight simultaneously
train_pg.broadcast(weight_tensor, src=0)
ray.get(handle)

# Verify weights changed
weights_changed = ray.get(llm.collective_rpc.remote("check_weights_changed"))
print(f"Weights updated: {all(weights_changed)}")
</syntaxhighlight>

=== Using ColocateWorkerExtension ===
<syntaxhighlight lang="python">
# Create colocated LLM
llm = ray.remote(num_gpus=0)(MyLLM).remote(
    model="facebook/opt-125m",
    worker_extension_cls="rlhf_utils.ColocateWorkerExtension",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.4,
    bundle_indices=[0, 1],
)

# Get device UUIDs from workers
device_ids = ray.get(llm.collective_rpc.remote("report_device_id", args=tuple()))
print(f"Workers on devices: {device_ids}")

# Create ZMQ handles from training actors
zmq_handles = {}
for actor in training_actors:
    zmq_handles.update(ray.get(actor.get_zmq_handles.remote()))

# Update weights via CUDA IPC
ray.get(
    [actor.update_weights.remote() for actor in training_actors]
    +
    [llm.collective_rpc.remote("update_weights_from_ipc", args=(zmq_handles,))]
)

# Verify
assert ray.get(llm.collective_rpc.remote("check_weights_changed", args=tuple()))
</syntaxhighlight>

=== Implementing Custom Worker Extension ===
<syntaxhighlight lang="python">
# In custom_extension.py
class CustomWorkerExtension:
    """Custom worker extension with additional methods."""

    def get_model_stats(self):
        """Return model statistics."""
        num_params = sum(
            p.numel() for p in self.model_runner.model.parameters()
        )
        return {
            "num_parameters": num_params,
            "device": str(self.device),
            "dtype": str(next(self.model_runner.model.parameters()).dtype),
        }

    def update_weight_custom(self, name, tensor_dict):
        """Custom weight update with preprocessing."""
        # Custom preprocessing
        weight = preprocess_weight(tensor_dict)

        # Load into model
        self.model_runner.model.load_weights(weights=[(name, weight)])

    def export_activations(self, layer_name):
        """Export intermediate activations for analysis."""
        # Hook into model forward pass
        activations = []

        def hook(module, input, output):
            activations.append(output.detach().cpu())

        # Register hook and return activations
        handle = register_hook(self.model_runner.model, layer_name, hook)
        return activations

# Usage
llm = LLM(
    model="facebook/opt-125m",
    worker_extension_cls="custom_extension.CustomWorkerExtension",
)

stats = llm.collective_rpc("get_model_stats", args=tuple())
</syntaxhighlight>

=== CUDA IPC Helper Usage ===
<syntaxhighlight lang="python">
from rlhf_utils import rebuild_ipc
from torch.multiprocessing.reductions import reduce_tensor
import torch

# Sender process
buffer = torch.randn(1024, 1024, device="cuda:0")
ipc_handle = reduce_tensor(buffer)

# Send handle via ZMQ or other IPC mechanism
send_to_receiver(ipc_handle)

# Receiver process (different CUDA_VISIBLE_DEVICES)
ipc_handle = receive_from_sender()

# Rebuild tensor with device ID remapping
local_device_id = 1  # Map to local visible device
buffer_local = rebuild_ipc(ipc_handle, device_id=local_device_id)

print(f"Received tensor shape: {buffer_local.shape}")
</syntaxhighlight>

=== Weight Update Pattern ===
<syntaxhighlight lang="python">
def sync_weights_to_inference(train_model, inference_llm, update_group):
    """Synchronize training weights to inference engine."""

    for name, param in train_model.named_parameters():
        # Prepare update on inference side
        dtype_name = str(param.dtype).split(".")[-1]
        handle = inference_llm.collective_rpc.remote(
            "update_weight",
            args=(name, dtype_name, param.shape)
        )

        # Broadcast from training side
        update_group.broadcast(
            param,
            src=0,
            stream=torch.cuda.current_stream()
        )

        # Wait for completion
        ray.get(handle)

    # Verify all weights updated
    all_updated = ray.get(
        inference_llm.collective_rpc.remote("check_weights_changed")
    )
    assert all(all_updated), "Weight synchronization failed"
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[related_to::Implementation:vllm-project_vllm_RLHF_Example]]
* [[related_to::Implementation:vllm-project_vllm_RLHF_Colocate_Example]]
* [[related_to::Implementation:vllm-project_vllm_RLHF_Online_Quant_Example]]
