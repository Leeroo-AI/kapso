= LLM Initialization with Speculative Decoding =

== Metadata ==
{| class="wikitable"
|-
! Field !! Value
|-
| Knowledge Sources || [https://arxiv.org/abs/2401.15077 EAGLE], [https://arxiv.org/abs/2404.19124 MLP Speculator], vLLM Source Code
|-
| Domains || LLM Engine, Offline Inference, Speculative Decoding
|-
| Last Updated || 2025-12-17
|}

== Overview ==

The <code>LLM</code> class initialization with speculative decoding enables accelerated inference by configuring the engine with draft models or draft-free speculation methods. This implementation realizes the [[implements::Principle:vllm-project_vllm_speculative_engine_init]] principle.

== Code Reference ==

=== Source Location ===
<syntaxhighlight lang="text">
File: vllm/entrypoints/llm.py
Class: LLM
Method: __init__
Lines: 93-400 (approximately)
</syntaxhighlight>

=== Signature ===
<syntaxhighlight lang="python">
class LLM:
    def __init__(
        self,
        model: str,
        *,
        runner: RunnerOption = "auto",
        tokenizer: str | None = None,
        tokenizer_mode: TokenizerMode | str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: ModelDType | str = "auto",
        quantization: str | None = None,
        revision: str | None = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        max_model_len: int | None = None,
        speculative_config: dict | SpeculativeConfig | None = None,
        # ... additional parameters
        **kwargs: Any,
    ) -> None:
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.config import SpeculativeConfig  # For advanced usage
</syntaxhighlight>

== Description ==

The <code>LLM</code> class provides offline (batch) inference with optional speculative decoding. When <code>speculative_config</code> is provided, the engine initializes additional components for draft token generation and verification.

=== Key Parameters ===

* '''model''': Target model path or identifier
* '''speculative_config''': Dictionary or SpeculativeConfig instance configuring speculation
* '''tensor_parallel_size''': TP for target model
* '''gpu_memory_utilization''': Memory allocation ratio (must account for draft model)
* '''enforce_eager''': Disable CUDA graphs (may be required for some methods)

== Input/Output Contract ==

=== Input: speculative_config Parameter ===

{| class="wikitable"
! Field !! Type !! Required !! Description
|-
| method || str || Conditional || Speculative method: "ngram", "eagle", "eagle3", "mlp_speculator", "suffix", "mtp"
|-
| model || str || For draft-based || Path to draft model or EAGLE head
|-
| num_speculative_tokens || int || Yes || Number of tokens to speculate (> 0)
|-
| draft_tensor_parallel_size || int || Optional || TP for draft model (1 or same as target)
|-
| prompt_lookup_max || int || For ngram || Maximum n-gram window
|-
| prompt_lookup_min || int || For ngram || Minimum n-gram window
|}

=== Output ===

Returns an initialized <code>LLM</code> instance ready for:
* Generating text with <code>generate()</code>
* Chat completion with <code>chat()</code>
* Retrieving metrics with <code>get_metrics()</code>

== Usage Examples ==

=== Example 1: Basic N-gram Speculation ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

# Initialize with ngram speculation (no draft model needed)
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
    }
)

# Generate with speculation
prompts = ["The capital of France is", "Write a Python function to"]
sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
    print("-" * 50)
</syntaxhighlight>

=== Example 2: EAGLE Speculation with Metrics ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

# Initialize with EAGLE for better acceptance rates
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
    speculative_config={
        "method": "eagle",
        "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        "num_speculative_tokens": 3,
    },
    enforce_eager=False,
    gpu_memory_utilization=0.9,
)

# Generate text
prompts = ["Explain quantum computing in simple terms."]
sampling_params = SamplingParams(temperature=0.0, max_tokens=200)
outputs = llm.generate(prompts, sampling_params)

# Retrieve acceptance metrics
metrics = llm.get_metrics()

for metric in metrics:
    if metric.name == "vllm:spec_decode_num_accepted_tokens":
        print(f"Accepted tokens: {metric.value}")
    elif metric.name == "vllm:spec_decode_num_draft_tokens":
        print(f"Draft tokens: {metric.value}")

# Calculate acceptance rate
num_drafts = sum(m.value for m in metrics if m.name == "vllm:spec_decode_num_drafts")
num_accepted = sum(m.value for m in metrics if m.name == "vllm:spec_decode_num_accepted_tokens")
if num_drafts > 0:
    mean_acceptance_length = 1 + (num_accepted / num_drafts)
    print(f"Mean acceptance length: {mean_acceptance_length:.2f}")
</syntaxhighlight>

=== Example 3: EAGLE3 with Multi-GPU ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

# EAGLE3 with tensor parallelism on target model
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,  # Target model uses 4 GPUs
    speculative_config={
        "method": "eagle3",
        "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-70B",
        "num_speculative_tokens": 3,
        "draft_tensor_parallel_size": 1,  # EAGLE runs on single GPU
    },
    gpu_memory_utilization=0.85,
)

prompts = ["Write a detailed essay about climate change."]
outputs = llm.generate(prompts, SamplingParams(max_tokens=500))

for output in outputs:
    print(output.outputs[0].text)
</syntaxhighlight>

=== Example 4: MLP Speculator ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

# MLP speculator for balanced performance
llm = LLM(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    speculative_config={
        "model": "ibm-ai-platform/llama3-70b-accelerator",
        # num_speculative_tokens auto-detected from draft model config
        "draft_tensor_parallel_size": 1,
    },
    gpu_memory_utilization=0.85,
)

prompts = ["def factorial(n):"]
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=150,
)
outputs = llm.generate(prompts, sampling_params)
</syntaxhighlight>

=== Example 5: Suffix Decoding for Repetitive Tasks ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

# Suffix decoding excels at repetitive patterns
# Requires: pip install arctic-inference
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "suffix",
        "num_speculative_tokens": 32,  # Maximum, adjusted dynamically
        "suffix_decoding_max_spec_factor": 1.0,
        "suffix_decoding_min_token_prob": 0.1,
        "suffix_decoding_max_cached_requests": 1000,
    }
)

# Example: Code refactoring (repetitive patterns)
prompts = [
    "Refactor this code to use list comprehension: result = []\nfor i in range(10):\n    result.append(i*2)"
]
outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=200))
</syntaxhighlight>

=== Example 6: Conditional Speculation ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

# Disable speculation when batch size is large
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
        "disable_by_batch_size": 8,  # Disable if batch > 8
    }
)

# Large batch - speculation may be disabled automatically
large_batch = ["Prompt " + str(i) for i in range(10)]
outputs = llm.generate(large_batch, SamplingParams(max_tokens=50))
</syntaxhighlight>

== Design Details ==

=== Initialization Workflow ===

1. Parse <code>speculative_config</code> (dict → SpeculativeConfig)
2. Create <code>EngineArgs</code> with speculative configuration
3. Initialize <code>LLMEngine</code> with both target and draft configs
4. Load target model weights
5. Load draft model weights (if applicable)
6. Initialize proposer (EAGLE, Ngram, MLP, etc.)
7. Set up rejection sampler for verification
8. Allocate memory for both models
9. Initialize CUDA graphs (if not disabled)

=== Memory Management ===

The engine allocates memory for:
* Target model weights and activations
* Draft model weights (EAGLE, MLP) or minimal state (ngram)
* Separate KV caches for target and draft
* Tree attention metadata (for EAGLE)
* Rejection sampling state

Memory calculation:
<syntaxhighlight lang="text">
total_memory = target_model_memory + draft_model_memory + kv_cache_memory
gpu_memory_utilization must account for all components
</syntaxhighlight>

=== Compatibility Matrix ===

{| class="wikitable"
! Method !! Draft Model !! TP Support !! CUDA Graph !! Notes
|-
| ngram || No || Full || Yes || Zero-cost, pattern-based
|-
| eagle || Yes || Draft TP=1 || Limited || Best acceptance rate
|-
| eagle3 || Yes || Draft TP=1 || Limited || Improved over EAGLE
|-
| mlp_speculator || Yes || Draft TP=1 || Yes || Lightweight speculator
|-
| suffix || No || Full || Yes || Requires arctic-inference
|-
| mtp || No (built-in) || Full || Limited || Native model support
|}

== Performance Considerations ==

=== When to Use Speculative Decoding ===

* '''Good Cases''': Long generations, repetitive patterns, code, low-medium batch sizes
* '''Poor Cases''': Very short generations, high batch sizes, extremely creative tasks
* '''Memory-Bound''': Most beneficial when inference is memory-bound (typical for large models)

=== Expected Speedups ===

* '''N-gram''': 1.2-1.5x for repetitive content
* '''EAGLE''': 1.5-2.3x average (varies by task)
* '''MLP Speculator''': 1.3-1.8x with minimal overhead
* '''Suffix''': 1.5-3x for highly repetitive tasks

== Related Pages ==

* [[implements::Principle:vllm-project_vllm_speculative_engine_init]]
* [[implemented_by::vllm-project_vllm_LLM_generate_spec]]
* [[implements::Principle:vllm-project_vllm_speculative_engine_init]]
* vLLM Generate Method
* Metrics Retrieval
