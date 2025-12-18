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

Concrete tool for executing text generation with speculative decoding acceleration (transparent API, accelerated internally).

=== Description ===

`LLM.generate()` on a speculative-enabled LLM provides the same interface as standard generation. Internally:
- Draft mechanism proposes K tokens
- Target model verifies all K in parallel
- Accepted tokens are output, rejected trigger re-sampling
- Process repeats until completion

The API is transparent—users call `generate()` normally.

=== Usage ===

Generate with speculation when:
- Running inference on speculative-enabled LLM
- No code changes needed from standard generation
- Metrics can be checked for optimization tuning

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project/vllm]
* '''File:''' vllm/entrypoints/llm.py
* '''Lines:''' L365-434

=== Signature ===
<syntaxhighlight lang="python">
# Same API as standard generation
def generate(
    self,
    prompts: PromptType | Sequence[PromptType],
    sampling_params: SamplingParams | None = None,
    **kwargs,
) -> list[RequestOutput]:
    """Generate with speculative decoding (transparent)."""
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
| prompts || PromptType || Yes || Input prompts (same as standard)
|-
| sampling_params || SamplingParams || No || Generation parameters (same as standard)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| outputs || list[RequestOutput] || Generated text (same format as standard)
|}

== Usage Examples ==

=== Basic Speculative Generation ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

# Initialize with speculation
llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
    },
)

# Generate - same API as non-speculative
outputs = llm.generate(
    ["What is machine learning?"],
    SamplingParams(max_tokens=100, temperature=0.7),
)

print(outputs[0].outputs[0].text)
</syntaxhighlight>

=== Batch Generation ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    speculative_config={"method": "ngram", "num_speculative_tokens": 5},
)

# Batch of prompts - speculation benefits batch processing
prompts = [
    "Explain Python decorators:",
    "What is a REST API?",
    "Describe machine learning:",
    "How do databases work?",
]

outputs = llm.generate(prompts, SamplingParams(max_tokens=150))

for prompt, output in zip(prompts, outputs):
    print(f"Q: {prompt[:30]}...")
    print(f"A: {output.outputs[0].text[:100]}...\n")
</syntaxhighlight>

=== Code Completion (High Speculation Benefit) ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

# N-gram speculation works well for code (repetitive patterns)
llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 5,
        "prompt_lookup_min": 2,
    },
)

code_prompt = '''
def fibonacci(n):
    """Calculate fibonacci sequence."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    """Calculate factorial."""
'''

outputs = llm.generate(
    [code_prompt],
    SamplingParams(max_tokens=100, temperature=0, stop=["\n\n"]),
)

print(outputs[0].outputs[0].text)
</syntaxhighlight>

=== Comparing Output Quality ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

# Speculation should NOT change output distribution (for temperature=0)
llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    speculative_config={"method": "ngram", "num_speculative_tokens": 5},
)

# Greedy decoding should be deterministic
params = SamplingParams(temperature=0, max_tokens=50)

output1 = llm.generate(["Hello"], params)
output2 = llm.generate(["Hello"], params)

# With temperature=0, outputs should be identical
assert output1[0].outputs[0].text == output2[0].outputs[0].text
print("✓ Speculative decoding maintains determinism")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Speculative_Generation]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
