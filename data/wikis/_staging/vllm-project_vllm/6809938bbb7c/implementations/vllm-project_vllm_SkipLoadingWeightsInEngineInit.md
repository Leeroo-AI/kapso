{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::Offline Inference]], [[domain::Model Loading]], [[domain::Weight Management]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Demonstrates how to initialize vLLM with dummy weights and later reload real weights dynamically, enabling faster startup for testing or staged initialization workflows.

=== Description ===
This example shows an advanced pattern where the LLM engine is created with <code>load_format="dummy"</code>, which initializes model architecture with random/dummy weights instead of loading actual checkpoint weights. The real weights can then be loaded later using the <code>collective_rpc()</code> mechanism to update the load configuration and reload weights in place.

This approach is useful for scenarios where you want to:
* Quickly test model architecture or API integration without waiting for weight loading
* Stage initialization to separate setup from weight loading
* Implement custom weight loading logic or preprocessing
* Reduce startup time for development and testing iterations

=== Usage ===
Use this approach when:
* Developing or testing code that doesn't require correct model outputs
* Implementing custom weight management or caching systems
* Prototyping with model architectures before obtaining full checkpoints
* Building systems that dynamically switch between model weights
* Optimizing startup sequences for specific deployment patterns

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/skip_loading_weights_in_engine_init.py examples/offline_inference/skip_loading_weights_in_engine_init.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Run the example
python examples/offline_inference/skip_loading_weights_in_engine_init.py

# The script will:
# 1. Initialize with dummy weights (fast)
# 2. Generate nonsensical outputs
# 3. Reload real weights
# 4. Generate sensible outputs
</syntaxhighlight>

== Key Concepts ==

=== Dummy Weight Loading ===
The <code>load_format="dummy"</code> option:
* Initializes model layers with random weights (typically from PyTorch's default initialization)
* Skips reading checkpoint files from disk
* Dramatically reduces initialization time (seconds vs. minutes)
* Maintains correct model architecture and shapes
* Produces meaningless outputs until real weights are loaded

=== Collective RPC Mechanism ===
The <code>collective_rpc()</code> method enables runtime control of distributed workers:
* Sends commands to all workers in the engine
* Supports configuration updates and method invocations
* Works across tensor-parallel and pipeline-parallel ranks
* Enables dynamic reconfiguration without restarting

=== Two-Phase Initialization ===
The example demonstrates a two-phase pattern:

'''Phase 1: Fast startup with dummy weights'''
* Creates LLM instance in seconds
* Allows API testing, validation, warmup
* Generates outputs (though incorrect)

'''Phase 2: Dynamic weight loading'''
* Updates load configuration to "auto"
* Calls <code>reload_weights()</code> to load real checkpoint
* Model now produces correct outputs

== Usage Examples ==

=== Basic Dummy-to-Real Loading ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

prompts = ["Hello, my name is", "The capital of France is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Initialize with dummy weights (very fast)
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    load_format="dummy",
    enforce_eager=True,
    tensor_parallel_size=4,
)

# Outputs will be nonsensical
outputs = llm.generate(prompts, sampling_params)
print("Dummy outputs:", outputs[0].outputs[0].text)

# Load real weights dynamically
llm.collective_rpc(
    "update_config",
    args=({"load_config": {"load_format": "auto"}},)
)
llm.collective_rpc("reload_weights")

# Now outputs make sense
outputs = llm.generate(prompts, sampling_params)
print("Real outputs:", outputs[0].outputs[0].text)
</syntaxhighlight>

=== Testing Without Checkpoints ===
<syntaxhighlight lang="python">
# Use dummy weights for API testing
llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    load_format="dummy",
    tensor_parallel_size=2,
)

# Test API surface without waiting for weight loading
try:
    outputs = llm.generate(["test prompt"], sampling_params)
    print("API working, outputs shape correct")
except Exception as e:
    print(f"API error: {e}")

# Later load real weights if needed
if need_real_inference:
    llm.collective_rpc("update_config", args=({"load_config": {"load_format": "auto"}},))
    llm.collective_rpc("reload_weights")
</syntaxhighlight>

=== Staged Initialization for Services ===
<syntaxhighlight lang="python">
# Fast startup allows quick health check responses
llm = LLM(
    model="mistralai/Mistral-7B-v0.1",
    load_format="dummy",
)

# Signal service is "ready" (architecture initialized)
health_status = "initializing"

# Load weights in background or on-demand
def load_real_weights():
    llm.collective_rpc("update_config", args=({"load_config": {"load_format": "auto"}},))
    llm.collective_rpc("reload_weights")
    health_status = "ready"

# Can start serving requests before weights are loaded
# (or defer to staged loading)
import threading
loading_thread = threading.Thread(target=load_real_weights)
loading_thread.start()
</syntaxhighlight>

=== Custom Weight Loading ===
<syntaxhighlight lang="python">
# Initialize with dummy weights
llm = LLM(
    model="bigscience/bloom-7b1",
    load_format="dummy",
    tensor_parallel_size=8,
)

# Implement custom weight loading logic
def custom_load_weights():
    # Example: Load from S3, apply custom preprocessing, etc.
    weights = download_from_custom_source()
    preprocess_weights(weights)

    # Update to use your custom weights
    llm.collective_rpc("update_config", args=({"load_config": {"load_format": "auto"}},))
    llm.collective_rpc("reload_weights")

custom_load_weights()
</syntaxhighlight>

== Configuration Requirements ==

=== enforce_eager Mode ===
The example uses <code>enforce_eager=True</code> because:
* CUDA graphs may cache dummy weights
* Reloading weights requires re-capturing graphs
* Eager mode ensures immediate effect of weight updates
* Production systems may need additional handling for CUDA graphs

=== Tensor Parallelism ===
The pattern works with distributed setups:
* <code>collective_rpc()</code> broadcasts to all workers
* Each TP rank reloads its shard of weights
* Synchronization ensures consistent state across ranks

== Performance Characteristics ==

=== Initialization Time Comparison ===
For a 7B parameter model with TP=4:
* '''Standard loading''': 30-60 seconds
* '''Dummy loading''': 2-5 seconds
* '''Reload time''': 30-60 seconds (same as initial load)

Total time is similar, but the pattern enables:
* Faster initial readiness signal
* Staged or deferred weight loading
* Testing before downloading full checkpoints

=== Memory Usage ===
* Dummy weights consume the same memory as real weights
* No memory savings during the dummy phase
* Useful for memory validation without checkpoint access

== Limitations and Considerations ==

=== Output Quality ===
* Dummy weights produce completely random/meaningless outputs
* Do not use dummy outputs for any production or evaluation purposes
* Ensure weight reloading completes before serving real requests

=== Compatibility ===
* Requires <code>enforce_eager=True</code> or careful CUDA graph management
* May not work with all quantization methods during reload
* Some features may behave differently with dummy weights

=== Use Case Fit ===
Best for:
* Development and testing workflows
* CI/CD pipelines that validate without inference quality checks
* Custom weight loading implementations

Not suitable for:
* Production serving (adds complexity)
* Scenarios where single-phase initialization suffices
* Models where initialization time is already negligible

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[related::Implementation:vllm-project_vllm_SaveShardedState]]
* [[related::Concept:vllm-project_vllm_Model_Loading]]
* [[related::Concept:vllm-project_vllm_Tensor_Parallelism]]
* [[related::API:vllm-project_vllm_LLM]]
