{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::RLHF]], [[domain::Distributed Training]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
This example demonstrates reinforcement learning from human feedback (RLHF) integration with vLLM using Ray for distributed execution.

=== Description ===
The RLHF example shows how to integrate vLLM inference with RLHF training workflows using Ray for process coordination. The pattern separates training (Hugging Face Transformers on GPU 0) from inference (vLLM with tensor parallelism on GPUs 1-2), enabling efficient RLHF training loops. The example demonstrates the complete cycle: generating rollouts with vLLM, updating weights on the training model, and broadcasting updates to the inference engine via collective communication. This architecture enables production RLHF systems where inference throughput is critical for collecting training data.

=== Usage ===
Use this pattern when implementing RLHF training pipelines, building PPO or similar RL-based fine-tuning systems, requiring fast inference for generating training rollouts, or separating training and inference to different hardware resources. For production systems with multiple replicas, see OpenRLHF.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/rlhf.py examples/offline_inference/rlhf.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
python examples/offline_inference/rlhf.py
</syntaxhighlight>

Note: Requires 3 GPUs (1 for training, 2 for inference).

== Key Concepts ==

=== Architecture Overview ===
The example demonstrates separated training and inference:
* '''Training Model''': Hugging Face Transformers on GPU 0
* '''Inference Engine''': vLLM with TP=2 on GPUs 1-2
* '''Coordination''': Ray placement groups for process management
* '''Communication''': Collective RPC for weight synchronization

=== Ray Placement Groups ===
Placement groups reserve GPU resources for components:
* Create placement group with 2 GPU bundles for vLLM
* Each bundle gets 1 GPU and 0 CPUs (GPU-only)
* PlacementGroupSchedulingStrategy ensures proper placement
* Prevents resource conflicts between training and inference

=== Custom LLM Class ===
MyLLM extends vLLM to work with Ray placement:
* Removes Ray's CUDA_VISIBLE_DEVICES override
* Allows vLLM to manage device placement internally
* Required for proper multi-GPU coordination

=== Worker Extension ===
The WorkerExtension (from rlhf_utils.py) provides:
* '''init_weight_update_group()''': Initialize NCCL communication
* '''update_weight()''': Receive weight updates via broadcast
* '''check_weights_changed()''': Verify synchronization

=== Weight Update Flow ===
The example shows the complete RLHF update cycle:
1. Generate rollouts using current policy (vLLM inference)
2. Compute rewards and update training model (simulated as zeroing)
3. Establish communication group between processes
4. Broadcast updated weights from training to inference
5. Verify weights synchronized correctly
6. Generate with updated policy

== Usage Examples ==

=== Ray and Placement Group Setup ===
<syntaxhighlight lang="python">
import os
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

# Load training model on GPU 0
from transformers import AutoModelForCausalLM
train_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
train_model.to("cuda:0")

# Initialize Ray for inference on GPUs 1-2
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
ray.init()

# Create placement group for vLLM
pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * 2)
ray.get(pg_inference.ready())

scheduling_inference = PlacementGroupSchedulingStrategy(
    placement_group=pg_inference,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=0,
)
</syntaxhighlight>

=== Custom LLM with Worker Extension ===
<syntaxhighlight lang="python">
from vllm import LLM

class MyLLM(LLM):
    """Configure vLLM worker for Ray placement group execution."""
    def __init__(self, *args, **kwargs):
        # Remove Ray's CUDA_VISIBLE_DEVICES override
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        super().__init__(*args, **kwargs)

# Launch vLLM inference engine
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
</syntaxhighlight>

=== Generating Rollouts ===
<syntaxhighlight lang="python">
from vllm import SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0)

# Generate with current policy
outputs = ray.get(llm.generate.remote(prompts, sampling_params))

for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
    print("-" * 50)
</syntaxhighlight>

=== Weight Synchronization ===
<syntaxhighlight lang="python">
import torch
from vllm.utils.network_utils import get_ip, get_open_port
from rlhf_utils import stateless_init_process_group

# Establish communication group
master_address = get_ip()
master_port = get_open_port()

# Initialize inference side (rank 1-2 in group of 3)
handle = llm.collective_rpc.remote(
    "init_weight_update_group",
    args=(master_address, master_port, 1, 3)
)

# Initialize training side (rank 0 in group of 3)
model_update_group = stateless_init_process_group(
    master_address, master_port, 0, 3, torch.device("cuda:0")
)
ray.get(handle)

# Simulate training update (zero out weights)
for name, p in train_model.named_parameters():
    p.data.zero_()

# Broadcast updates to inference engine
for name, p in train_model.named_parameters():
    dtype_name = str(p.dtype).split(".")[-1]
    handle = llm.collective_rpc.remote(
        "update_weight",
        args=(name, dtype_name, p.shape)
    )
    model_update_group.broadcast(p, src=0, stream=torch.cuda.current_stream())
    ray.get(handle)

# Verify synchronization
assert all(ray.get(llm.collective_rpc.remote("check_weights_changed")))
</syntaxhighlight>

=== Complete RLHF Loop ===
<syntaxhighlight lang="python">
def rlhf_training_step(train_model, llm, prompts):
    """Single RLHF training iteration."""
    # 1. Generate rollouts
    sampling_params = SamplingParams(temperature=0.7, max_tokens=50)
    outputs = ray.get(llm.generate.remote(prompts, sampling_params))

    # 2. Compute rewards (placeholder)
    rewards = compute_rewards(outputs)

    # 3. Update training model with RL objective (e.g., PPO)
    update_policy(train_model, outputs, rewards)

    # 4. Synchronize weights to inference engine
    synchronize_weights(train_model, llm)

    return outputs, rewards

# Training loop
for iteration in range(num_iterations):
    outputs, rewards = rlhf_training_step(
        train_model, llm, training_prompts
    )
    print(f"Iteration {iteration}: avg_reward={sum(rewards)/len(rewards)}")
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[related_to::Implementation:vllm-project_vllm_RLHF_Colocate_Example]]
* [[related_to::Implementation:vllm-project_vllm_RLHF_Online_Quant_Example]]
* [[related_to::Implementation:vllm-project_vllm_RLHF_Utils]]
