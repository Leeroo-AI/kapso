{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::RLHF]], [[domain::Quantization]], [[domain::FP8]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
This example demonstrates RLHF integration with FP8 dynamic quantization for memory-efficient inference during RL training.

=== Description ===
The RLHF online quantization example extends the standard RLHF pattern by adding FP8 dynamic quantization to the inference engine, reducing memory usage during rollout generation. The example uses torchao's Float8DynamicActivationFloat8WeightConfig to quantize both activations and weights dynamically, enabling larger batch sizes or larger models within the same memory budget. The quantization configuration is serialized to JSON and passed via hf_overrides, demonstrating how to integrate quantization into RLHF workflows. This is particularly valuable for scaling RLHF training with limited GPU resources.

=== Usage ===
Use this pattern when GPU memory is constrained during RLHF training, you want to increase rollout batch sizes for better sample efficiency, need to fit larger models for inference during RL training, or want to optimize inference throughput without sacrificing training model precision.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/rlhf_online_quant.py examples/offline_inference/rlhf_online_quant.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
python examples/offline_inference/rlhf_online_quant.py
</syntaxhighlight>

Note: Requires 3 GPUs (1 for training, 2 for quantized inference).

== Key Concepts ==

=== FP8 Dynamic Quantization ===
The example uses torchao for FP8 quantization:
* '''Float8DynamicActivationFloat8WeightConfig''': Both weights and activations in FP8
* '''granularity=PerRow()''': Row-wise quantization for better accuracy
* Dynamic: Computed per-batch rather than static calibration
* Memory savings: ~50% compared to FP16/BF16
* Throughput improvement: Faster inference on modern GPUs (H100, etc.)

=== Quantization Configuration ===
Configuration is serialized and passed to vLLM:
* '''config_to_dict()''': Convert torchao config to dictionary
* '''json.dumps()''': Serialize to JSON string
* '''hf_overrides["quantization_config_dict_json"]''': Pass to vLLM
* Alternative: Use serialized config files (see PR #23014)

=== Architecture Pattern ===
Same separation as standard RLHF:
* Training model: Full precision on GPU 0 (no quantization)
* Inference engine: FP8 quantized on GPUs 1-2 with TP=2
* Communication: Ray collective RPC for weight updates
* Training precision preserved for gradient accuracy

=== Weight Update Flow ===
Weight updates account for quantization:
1. Training model updated in full precision
2. Weights broadcast to inference engine in full precision
3. vLLM dynamically quantizes weights upon loading
4. Inference runs with FP8 quantized weights
5. Next training update uses full precision again

== Usage Examples ==

=== Creating Quantization Config ===
<syntaxhighlight lang="python">
import json
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    PerRow,
)
from torchao.core.config import config_to_dict

# Create FP8 quantization configuration
config = Float8DynamicActivationFloat8WeightConfig(
    granularity=PerRow()
)

# Serialize to JSON string
json_str = json.dumps(config_to_dict(config))

print(f"Quantization config: {json_str}")
</syntaxhighlight>

=== Initializing Quantized Inference Engine ===
<syntaxhighlight lang="python">
import os
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM

class MyLLM(LLM):
    """Configure vLLM worker for Ray placement group execution."""
    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        super().__init__(*args, **kwargs)

# Setup Ray and placement groups
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
ray.init()

pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * 2)
ray.get(pg_inference.ready())

scheduling_inference = PlacementGroupSchedulingStrategy(
    placement_group=pg_inference,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=0,
)

# Launch quantized vLLM inference engine
config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
json_str = json.dumps(config_to_dict(config))

llm = ray.remote(
    num_cpus=0,
    num_gpus=0,
    scheduling_strategy=scheduling_inference,
)(MyLLM).remote(
    model="facebook/opt-125m",
    hf_overrides={"quantization_config_dict_json": json_str},
    enforce_eager=True,
    worker_extension_cls="rlhf_utils.WorkerExtension",
    tensor_parallel_size=2,
    distributed_executor_backend="ray",
)
</syntaxhighlight>

=== Training Model Setup ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
import torch

# Load training model in full precision
train_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
train_model.to("cuda:0")

# Training model stays in full precision for accurate gradients
print(f"Training model dtype: {train_model.dtype}")
</syntaxhighlight>

=== Generating Quantized Rollouts ===
<syntaxhighlight lang="python">
from vllm import SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0)

# Generate with FP8 quantized model
outputs = ray.get(llm.generate.remote(prompts, sampling_params))

print("-" * 50)
for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
    print("-" * 50)
</syntaxhighlight>

=== Weight Synchronization with Quantization ===
<syntaxhighlight lang="python">
from vllm.utils.network_utils import get_ip, get_open_port
from rlhf_utils import stateless_init_process_group

# Setup communication
master_address = get_ip()
master_port = get_open_port()

handle = llm.collective_rpc.remote(
    "init_weight_update_group",
    args=(master_address, master_port, 1, 3)
)

model_update_group = stateless_init_process_group(
    master_address, master_port, 0, 3, torch.device("cuda:0")
)
ray.get(handle)

# Simulate training update (full precision)
for name, p in train_model.named_parameters():
    p.data.zero_()

# Broadcast full precision weights
for name, p in train_model.named_parameters():
    dtype_name = str(p.dtype).split(".")[-1]
    handle = llm.collective_rpc.remote(
        "update_weight",
        args=(name, dtype_name, p.shape)
    )
    # Broadcast in full precision
    model_update_group.broadcast(p, src=0, stream=torch.cuda.current_stream())
    ray.get(handle)
    # vLLM will quantize to FP8 internally

# Verify updates
assert all(ray.get(llm.collective_rpc.remote("check_weights_changed")))
</syntaxhighlight>

=== Memory Comparison ===
<syntaxhighlight lang="python">
import torch

def measure_inference_memory(model_name, use_quantization=False):
    """Measure GPU memory usage for inference."""
    torch.cuda.reset_peak_memory_stats()

    if use_quantization:
        config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
        json_str = json.dumps(config_to_dict(config))
        llm = LLM(
            model=model_name,
            hf_overrides={"quantization_config_dict_json": json_str},
            tensor_parallel_size=2,
        )
    else:
        llm = LLM(model=model_name, tensor_parallel_size=2)

    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Quantized={use_quantization}: {peak_memory:.2f}GB peak memory")

    return peak_memory

# Compare memory usage
fp16_memory = measure_inference_memory("facebook/opt-125m", use_quantization=False)
fp8_memory = measure_inference_memory("facebook/opt-125m", use_quantization=True)

print(f"Memory savings: {(1 - fp8_memory/fp16_memory)*100:.1f}%")
</syntaxhighlight>

=== Complete RLHF Training Loop ===
<syntaxhighlight lang="python">
def rlhf_training_iteration(train_model, quantized_llm, prompts):
    """Single RLHF training step with quantized inference."""

    # 1. Generate rollouts with FP8 quantized model
    sampling_params = SamplingParams(temperature=0.7, max_tokens=50)
    outputs = ray.get(quantized_llm.generate.remote(prompts, sampling_params))

    # 2. Compute rewards
    rewards = compute_rewards(outputs)

    # 3. Update training model (full precision)
    loss = compute_ppo_loss(train_model, outputs, rewards)
    loss.backward()
    optimizer.step()

    # 4. Broadcast updated weights (full precision -> FP8)
    sync_weights_to_quantized_engine(train_model, quantized_llm)

    return outputs, rewards

# Main training loop
for step in range(num_training_steps):
    outputs, rewards = rlhf_training_iteration(
        train_model, llm, training_prompts
    )
    print(f"Step {step}: reward={sum(rewards)/len(rewards):.3f}")
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[related_to::Implementation:vllm-project_vllm_RLHF_Example]]
* [[related_to::Implementation:vllm-project_vllm_RLHF_Utils]]
