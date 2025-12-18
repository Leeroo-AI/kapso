{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::Offline Inference]], [[domain::Performance]], [[domain::Profiling]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Demonstrates basic performance profiling of vLLM inference using PyTorch's built-in profiler to analyze execution time and resource usage.

=== Description ===
This example shows how to enable and use the PyTorch profiler with vLLM to capture detailed performance metrics during inference. The profiler records kernel execution times, memory operations, and CPU activity, generating timeline traces that can be visualized with tools like Chrome's trace viewer or TensorBoard.

The script wraps a simple generation task with <code>start_profile()</code> and <code>stop_profile()</code> calls, directing output to a specified directory. This provides insights into where time is spent during inference, helping identify bottlenecks and optimization opportunities.

=== Usage ===
Use this approach when:
* Investigating performance issues or unexpected slowdowns
* Comparing different model configurations or settings
* Optimizing custom kernels or operations
* Understanding GPU utilization patterns
* Preparing performance reports or benchmarks
* Debugging multi-process or distributed execution

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/simple_profiling.py examples/offline_inference/simple_profiling.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Run with default profiling configuration
python examples/offline_inference/simple_profiling.py

# View the generated trace files
ls -lh ./vllm_profile/

# Open in Chrome tracing (chrome://tracing)
# Load the JSON trace file from ./vllm_profile/
</syntaxhighlight>

== Key Concepts ==

=== PyTorch Profiler Integration ===
vLLM provides a <code>profiler_config</code> parameter that accepts:
* '''profiler''': The profiler backend to use (e.g., "torch")
* '''torch_profiler_dir''': Output directory for trace files
* Additional profiler-specific options

The profiler captures:
* CUDA kernel launches and execution times
* Memory allocation and transfer operations
* CPU function calls and overhead
* Inter-process communication (for distributed execution)

=== Profiling Workflow ===
The standard profiling pattern:
1. Configure LLM with <code>profiler_config</code> dictionary
2. Call <code>llm.start_profile()</code> to begin recording
3. Execute the workload to profile
4. Call <code>llm.stop_profile()</code> to finalize traces
5. Wait briefly for background processes to finish writing
6. Analyze trace files with visualization tools

=== Output Files ===
The profiler generates:
* '''JSON trace files''': Chrome tracing format for timeline visualization
* '''Per-rank traces''': Separate files for each process in distributed setups
* '''Metadata''': Information about the profiling session

== Usage Examples ==

=== Basic Profiling ===
<syntaxhighlight lang="python">
import time
from vllm import LLM, SamplingParams

prompts = ["Hello, my name is", "The capital of France is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create LLM with profiler enabled
llm = LLM(
    model="facebook/opt-125m",
    tensor_parallel_size=1,
    profiler_config={
        "profiler": "torch",
        "torch_profiler_dir": "./vllm_profile",
    },
)

# Start profiling
llm.start_profile()

# Run inference
outputs = llm.generate(prompts, sampling_params)

# Stop profiling
llm.stop_profile()

# Print results
for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")

# Wait for background profiler to finish
time.sleep(10)
</syntaxhighlight>

=== Profiling with Tensor Parallelism ===
<syntaxhighlight lang="python">
# Profile distributed inference
llm = LLM(
    model="meta-llama/Llama-3.1-70B",
    tensor_parallel_size=4,
    profiler_config={
        "profiler": "torch",
        "torch_profiler_dir": "./vllm_profile_tp4",
    },
)

llm.start_profile()
outputs = llm.generate(long_prompts, sampling_params)
llm.stop_profile()

# Will generate 4 trace files, one per TP rank
time.sleep(10)
</syntaxhighlight>

=== Profiling Specific Operations ===
<syntaxhighlight lang="python">
# Profile only the prefill phase
llm = LLM(
    model="mistralai/Mistral-7B-v0.1",
    profiler_config={
        "profiler": "torch",
        "torch_profiler_dir": "./prefill_profile",
    },
)

# Profile just prefill by using max_tokens=1
llm.start_profile()
outputs = llm.generate(
    prompts,
    SamplingParams(max_tokens=1, temperature=0.0)
)
llm.stop_profile()
time.sleep(10)
</syntaxhighlight>

== Analyzing Profiler Output ==

=== Chrome Tracing ===
To visualize the traces:
1. Open Chrome browser and navigate to <code>chrome://tracing</code>
2. Click "Load" and select the JSON trace file
3. Use WASD keys to navigate the timeline
4. Click on operations to see detailed timing information
5. Look for gaps, long operations, or synchronization issues

=== TensorBoard Integration ===
<syntaxhighlight lang="bash">
# Launch TensorBoard with profiler plugin
tensorboard --logdir=./vllm_profile

# Open browser to http://localhost:6006
# Navigate to the "Profile" tab
</syntaxhighlight>

=== Key Metrics to Examine ===
* '''Kernel execution time''': Time spent in CUDA kernels
* '''Memory operations''': Data transfers between CPU and GPU
* '''CPU overhead''': Python and C++ function call overhead
* '''Idle time''': Gaps indicating synchronization or scheduling delays
* '''Multi-GPU communication''': NCCL operations for tensor/pipeline parallelism

== Performance Considerations ==

=== Profiling Overhead ===
* Profiling adds 10-30% overhead to execution time
* Use profiling for analysis only, not production serving
* Profile representative workloads, not trivial examples
* Consider profiling warm-up runs separately from steady-state

=== Multi-Process Profiling ===
The example includes a 10-second sleep after <code>stop_profile()</code> because:
* Background processes may still be writing trace data
* Distributed workers flush asynchronously
* Ensures complete trace files before process termination

=== Storage Requirements ===
* Trace files can be 10MB-1GB depending on workload duration
* Limit profiling to short representative runs
* Clean up old trace files regularly

== Common Profiling Patterns ==

=== Comparing Configurations ===
<syntaxhighlight lang="bash">
# Profile different quantization methods
python simple_profiling.py --quantization=none --output=./profile_fp16
python simple_profiling.py --quantization=awq --output=./profile_awq
python simple_profiling.py --quantization=gptq --output=./profile_gptq
</syntaxhighlight>

=== Benchmarking Changes ===
<syntaxhighlight lang="python">
# Profile before and after optimization
import os

for config in ["baseline", "optimized"]:
    os.environ["MY_CONFIG"] = config
    llm = LLM(
        model="...",
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": f"./profile_{config}",
        },
    )
    llm.start_profile()
    outputs = llm.generate(prompts, sampling_params)
    llm.stop_profile()
    time.sleep(10)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
