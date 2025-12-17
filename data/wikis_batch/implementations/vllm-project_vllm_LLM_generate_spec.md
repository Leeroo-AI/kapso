= LLM.generate() with Speculative Decoding =

== Metadata ==
{| class="wikitable"
|-
! Field !! Value
|-
| Knowledge Sources || [https://arxiv.org/abs/2302.01318 Speculative Sampling], [https://arxiv.org/abs/2401.15077 EAGLE], vLLM Generate API
|-
| Domains || Text Generation, Speculative Decoding, Batch Inference
|-
| Last Updated || 2025-12-17
|}

== Overview ==

The <code>LLM.generate()</code> method performs batch text generation with optional speculative decoding acceleration. When speculative_config is enabled, it orchestrates draft proposal, parallel verification, and acceptance/rejection to achieve faster inference while maintaining output quality.

== Code Reference ==

=== Source Location ===
<syntaxhighlight lang="text">
File: vllm/entrypoints/llm.py
Class: LLM
Method: generate()
Lines: ~600-900 (approximate)
</syntaxhighlight>

=== Signature ===
<syntaxhighlight lang="python">
def generate(
    self,
    prompts: list[PromptType] | PromptType | None = None,
    sampling_params: SamplingParams | list[SamplingParams] | None = None,
    prompt_token_ids: list[list[int]] | None = None,
    use_tqdm: bool = True,
    lora_request: LoRARequest | list[LoRARequest] | None = None,
    prompt_adapter_request: PromptAdapterRequest | None = None,
    logits_processor: LogitsProcessor | list[LogitsProcessor] | None = None,
) -> list[RequestOutput]:
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt  # Optional for token-level control
</syntaxhighlight>

== Description ==

<code>generate()</code> implements the [[implements::Principle:vllm-project_vllm_speculative_generation]] principle by executing the speculative decoding loop when enabled. It manages the complete generation lifecycle including draft proposal, verification, acceptance, and fallback sampling.

=== Key Features ===

* '''Batch Processing''': Efficiently handles multiple prompts simultaneously
* '''Speculative Execution''': Transparently applies speculation when configured
* '''Progress Tracking''': Optional tqdm progress bar
* '''LoRA Support''': Compatible with LoRA adapters
* '''Flexible Input''': Accepts text, tokens, or multi-modal prompts

== Input/Output Contract ==

=== Input Parameters ===

{| class="wikitable"
! Parameter !! Type !! Required !! Description
|-
| prompts || list[str/TokensPrompt/...] || Conditional || Input prompts (text or tokens)
|-
| sampling_params || SamplingParams || Optional || Generation parameters (temperature, max_tokens, etc.)
|-
| prompt_token_ids || list[list[int]] || Conditional || Alternative to prompts (deprecated)
|-
| use_tqdm || bool || No || Show progress bar (default: True)
|-
| lora_request || LoRARequest || No || LoRA adapter to apply
|}

=== Output ===

Returns <code>list[RequestOutput]</code> where each <code>RequestOutput</code> contains:

{| class="wikitable"
! Field !! Type !! Description
|-
| prompt || str || Original prompt text
|-
| prompt_token_ids || list[int] || Tokenized prompt
|-
| outputs || list[CompletionOutput] || Generated completions (typically 1 unless n>1)
|-
| finished || bool || Whether generation completed
|}

Each <code>CompletionOutput</code> contains:

{| class="wikitable"
! Field !! Type !! Description
|-
| text || str || Generated text
|-
| token_ids || list[int] || Generated token IDs
|-
| cumulative_logprob || float || Sum of log probabilities
|-
| logprobs || list[dict] || Per-token logprob information (if requested)
|-
| finish_reason || str || Reason for completion ("stop", "length", etc.)
|}

== Usage Examples ==

=== Example 1: Basic Speculative Generation ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

# Initialize with ngram speculation
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
    }
)

# Generate with speculation
prompts = [
    "The capital of France is",
    "To define a function in Python, use",
]
sampling_params = SamplingParams(temperature=0.0, max_tokens=50)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
    print(f"Finish reason: {output.outputs[0].finish_reason}")
    print("-" * 50)
</syntaxhighlight>

=== Example 2: EAGLE with Temperature Control ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "eagle",
        "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        "num_speculative_tokens": 3,
    }
)

# Greedy decoding (temp=0) usually has highest acceptance rate
prompts = ["Explain neural networks"]
sampling_params = SamplingParams(
    temperature=0.0,  # Greedy for best speculation
    max_tokens=200,
    top_p=1.0,
)

outputs = llm.generate(prompts, sampling_params)
print(outputs[0].outputs[0].text)
</syntaxhighlight>

=== Example 3: Batch Generation with Mixed Lengths ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
    }
)

# Different prompt lengths - speculation adapts per request
prompts = [
    "Hi",  # Short prompt
    "Write a detailed explanation of quantum mechanics",  # Long prompt
    "def factorial(n):",  # Code prompt (good for ngram)
]

# Different max_tokens per prompt
sampling_params_list = [
    SamplingParams(temperature=0.0, max_tokens=10),   # Short generation
    SamplingParams(temperature=0.0, max_tokens=300),  # Long generation
    SamplingParams(temperature=0.0, max_tokens=100),  # Medium generation
]

outputs = llm.generate(prompts, sampling_params_list)

for i, output in enumerate(outputs):
    print(f"Prompt {i}: {output.prompt}")
    print(f"Generated tokens: {len(output.outputs[0].token_ids)}")
    print(f"Text: {output.outputs[0].text[:100]}...")
    print("-" * 50)
</syntaxhighlight>

=== Example 4: Speculation with TokensPrompt ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
    }
)

# Pre-tokenize for fine control
prompts_text = [
    "def add(a, b): return a + b\ndef subtract(a, b):",
    "The first law of thermodynamics states",
]

tokens_prompts = [
    TokensPrompt(prompt_token_ids=tokenizer.encode(p, add_special_tokens=False))
    for p in prompts_text
]

outputs = llm.generate(
    tokens_prompts,
    SamplingParams(temperature=0.0, max_tokens=100)
)

for output in outputs:
    print(f"Generated: {output.outputs[0].text}")
</syntaxhighlight>

=== Example 5: With Logprobs and Metrics ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "eagle",
        "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        "num_speculative_tokens": 3,
    }
)

prompts = ["The Eiffel Tower is located in"]
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=50,
    logprobs=5,  # Request top-5 logprobs per token
)

outputs = llm.generate(prompts, sampling_params)

# Access per-token logprobs
output = outputs[0]
for i, token_logprobs in enumerate(output.outputs[0].logprobs):
    print(f"Token {i}: {token_logprobs}")

# Get speculation metrics
metrics = llm.get_metrics()
for metric in metrics:
    if "spec_decode" in metric.name:
        print(f"{metric.name}: {metric.value}")
</syntaxhighlight>

=== Example 6: Streaming-Style with Progress Bar ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
    }
)

# Many prompts - show progress
prompts = [f"Explain concept {i}" for i in range(20)]
sampling_params = SamplingParams(temperature=0.0, max_tokens=100)

# use_tqdm=True shows progress (default)
outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

print(f"Generated {len(outputs)} outputs")
print(f"Total tokens: {sum(len(o.outputs[0].token_ids) for o in outputs)}")
</syntaxhighlight>

== Design Details ==

=== Execution Flow with Speculation ===

1. '''Input Processing'''
   - Parse prompts (text/tokens/multi-modal)
   - Create sampling parameters
   - Validate inputs

2. '''Engine Dispatch'''
   - Submit requests to LLMEngine
   - Engine creates internal request objects
   - Requests added to scheduler queue

3. '''Speculative Loop''' (per iteration)
   - '''Propose''': Generate k draft tokens using proposer (ngram/EAGLE/MLP)
   - '''Verify''': Forward pass through target model for all candidates
   - '''Accept/Reject''': Rejection sampling on target vs draft probabilities
   - '''Update''': Append accepted tokens, sample fallback if needed
   - '''Repeat''': Until max_tokens or stop condition

4. '''Output Collection'''
   - Collect finished requests
   - Decode token IDs to text
   - Return RequestOutput objects

=== Speculation Integration ===

When speculative_config is enabled:

* '''Proposer Selection''': Choose proposer based on method (NgramProposer, EagleProposer, etc.)
* '''Tree Attention''': For EAGLE, construct tree attention metadata
* '''Rejection Sampler''': Compare draft and target distributions
* '''Metrics Tracking''': Record acceptance rates, draft counts, etc.

=== Memory Management ===

* '''KV Cache''': Allocates blocks for both prompt and speculative tokens
* '''Block Recycling''': Rejected tokens' blocks returned to free pool
* '''Tree Structures''': EAGLE maintains tree of candidate paths
* '''Batch Coordination''': Scheduler manages memory across all requests

=== Performance Characteristics ===

==== Best Case (High Acceptance) ====
* Acceptance rate: 80-90%
* Speedup: 2-3x
* Example: Greedy generation, repetitive patterns

==== Average Case ====
* Acceptance rate: 50-70%
* Speedup: 1.5-2x
* Example: Creative text with some structure

==== Worst Case (Low Acceptance) ====
* Acceptance rate: <30%
* Speedup: <1.2x or slowdown
* Example: High temperature, very creative output
* Note: Can disable with disable_by_batch_size

=== Sampling Parameters Impact ===

{| class="wikitable"
! Parameter !! Impact on Speculation
|-
| temperature=0.0 || Best acceptance rate (greedy)
|-
| temperature>0.5 || Lower acceptance rate
|-
| top_k, top_p || May reduce acceptance if too restrictive
|-
| max_tokens || Longer generations amortize speculation overhead
|-
| presence_penalty || May affect acceptance patterns
|}

== Performance Optimization ==

=== Maximize Acceptance Rate ===
* Use greedy decoding (temperature=0) when possible
* Choose appropriate speculative method for task
* Provide context-rich prompts for better proposals
* Monitor metrics and adjust num_speculative_tokens

=== Minimize Overhead ===
* Use ngram for zero-cost proposals
* Keep draft models small (TP=1)
* Enable CUDA graphs where supported
* Set disable_by_batch_size for large batches

=== Memory Optimization ===
* Reduce num_speculative_tokens if memory constrained
* Use quantized draft models
* Monitor KV cache usage via metrics
* Adjust gpu_memory_utilization accordingly

== Common Patterns ==

=== Pattern: Adaptive Generation ===
<syntaxhighlight lang="python">
# Start with speculation, disable if not effective
llm = LLM(
    model="...",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
        "disable_by_batch_size": 16,  # Auto-disable for large batches
    }
)
</syntaxhighlight>

=== Pattern: Task-Specific Optimization ===
<syntaxhighlight lang="python">
# Code generation: ngram works well
code_llm = LLM(model="...", speculative_config={"method": "ngram", ...})

# Creative writing: EAGLE better
creative_llm = LLM(model="...", speculative_config={"method": "eagle", ...})
</syntaxhighlight>

== Related Pages ==

* [[implements::Principle:vllm-project_vllm_speculative_generation]]
* [[implemented_by::vllm-project_vllm_get_metrics]]
* [[implements::Principle:vllm-project_vllm_speculative_generation]]
* SamplingParams Documentation
* RequestOutput Structure
