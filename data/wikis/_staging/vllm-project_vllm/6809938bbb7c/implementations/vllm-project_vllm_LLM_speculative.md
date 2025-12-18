{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Paper|Speculative Decoding|https://arxiv.org/abs/2211.17192]]
|-
! Domains
| [[domain::NLP]], [[domain::Inference]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

Concrete tool for initializing an LLM instance with speculative decoding enabled for accelerated inference.

=== Description ===

Initializing `LLM` with `speculative_config` enables transparent speculative decoding. The resulting LLM:
- Loads both target and draft models (if applicable)
- Configures speculation pipeline
- Provides same `generate()` API (speculation is transparent)
- Accelerates inference without changing client code

=== Usage ===

Initialize speculative LLM when:
- Deploying latency-optimized inference
- Serving large models with speed requirements
- Running interactive applications (chat, completion)

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project/vllm]
* '''File:''' vllm/entrypoints/llm.py
* '''Lines:''' L190-337

=== Signature ===
<syntaxhighlight lang="python">
llm = LLM(
    model: str,
    speculative_config: dict,  # Enables speculation
    tensor_parallel_size: int = 1,
    dtype: str = "auto",
    enforce_eager: bool = False,
    **kwargs,
)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || str || Yes || Target model path
|-
| speculative_config || dict || Yes || Speculation configuration
|-
| tensor_parallel_size || int || No || GPU parallelism for target model
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| llm || LLM || Speculative-enabled LLM instance
|}

== Usage Examples ==

=== N-gram Speculative LLM ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

# Initialize with n-gram speculation
llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
    },
)

# Generate as normal - speculation is automatic
outputs = llm.generate(
    ["Write a Python function to calculate fibonacci:"],
    SamplingParams(max_tokens=200, temperature=0),
)

print(outputs[0].outputs[0].text)
</syntaxhighlight>

=== EAGLE Speculative LLM ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

# EAGLE speculation for maximum speedup
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "eagle",
        "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
        "num_speculative_tokens": 5,
    },
)

outputs = llm.generate(
    ["Explain the theory of relativity in simple terms:"],
    SamplingParams(max_tokens=300, temperature=0.7),
)
</syntaxhighlight>

=== Draft Model on Multi-GPU ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

# Large model with draft model speculation
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_config={
        "method": "draft_model",
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "num_speculative_tokens": 5,
        "draft_tensor_parallel_size": 1,  # Draft on 1 GPU
    },
    tensor_parallel_size=4,  # Target on 4 GPUs
    gpu_memory_utilization=0.85,
)

outputs = llm.generate(
    ["Write a detailed essay about climate change:"],
    SamplingParams(max_tokens=500),
)
</syntaxhighlight>

=== Comparing With/Without Speculation ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
import time

model_name = "meta-llama/Llama-3.2-1B-Instruct"
prompts = ["Write a story about a robot:"] * 10

# Without speculation
llm_standard = LLM(model=model_name)
start = time.time()
outputs_standard = llm_standard.generate(prompts, SamplingParams(max_tokens=200))
time_standard = time.time() - start

# With speculation
llm_spec = LLM(
    model=model_name,
    speculative_config={"method": "ngram", "num_speculative_tokens": 5},
)
start = time.time()
outputs_spec = llm_spec.generate(prompts, SamplingParams(max_tokens=200))
time_spec = time.time() - start

print(f"Standard: {time_standard:.2f}s")
print(f"Speculative: {time_spec:.2f}s")
print(f"Speedup: {time_standard/time_spec:.2f}x")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Speculative_Engine_Init]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
