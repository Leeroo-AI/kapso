{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Paper|Speculative Decoding|https://arxiv.org/abs/2211.17192]]
|-
! Domains
| [[domain::NLP]], [[domain::Inference]], [[domain::Configuration]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

Concrete tool for configuring speculative decoding parameters including method, draft model, and speculation depth.

=== Description ===

`speculative_config` is a dictionary passed to the LLM constructor that configures speculative decoding behavior:
- **method:** Which speculation strategy to use
- **model:** Draft model path (for draft_model/eagle methods)
- **num_speculative_tokens:** How many tokens to speculate per step
- **ngram settings:** Parameters for n-gram based speculation

=== Usage ===

Configure speculative decoding when:
- Setting up accelerated inference
- Tuning speculation depth
- Configuring draft models
- Optimizing acceptance rates

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project/vllm]
* '''File:''' vllm/config/speculative.py
* '''Lines:''' L54-150

=== Signature ===
<syntaxhighlight lang="python">
# speculative_config dict structure
speculative_config = {
    "method": str,                    # Required: speculation method
    "model": str | None,              # Draft model path (some methods)
    "num_speculative_tokens": int,    # Tokens to speculate (default: 5)

    # N-gram specific
    "prompt_lookup_max": int | None,  # Max n-gram size for lookup
    "prompt_lookup_min": int | None,  # Min n-gram size

    # Draft model specific
    "draft_tensor_parallel_size": int | None,  # TP for draft model
}
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm import LLM
# Pass speculative_config as dict to LLM constructor
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| method || str || Yes || Speculation method ("ngram", "eagle", etc.)
|-
| num_speculative_tokens || int || No || Tokens per speculation step (default: 5)
|-
| model || str || Conditional || Draft model path (for draft_model/eagle)
|-
| prompt_lookup_max || int || Conditional || Max n-gram size (for ngram)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| speculative_config || dict || Configuration dict for LLM constructor
|}

== Usage Examples ==

=== N-gram Speculation ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

# N-gram speculation (simplest setup)
llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
        "prompt_lookup_min": 2,
    },
)

# Use normally - speculation is transparent
outputs = llm.generate(["Hello, "], SamplingParams(max_tokens=100))
</syntaxhighlight>

=== EAGLE Speculation ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

# EAGLE speculation with draft heads
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "eagle",
        "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
        "num_speculative_tokens": 5,
    },
    tensor_parallel_size=1,
)

outputs = llm.generate(
    ["Explain quantum computing:"],
    SamplingParams(max_tokens=200),
)
</syntaxhighlight>

=== Draft Model Speculation ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

# Use smaller model as draft
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_config={
        "method": "draft_model",
        "model": "meta-llama/Llama-3.2-1B-Instruct",  # Draft
        "num_speculative_tokens": 5,
        "draft_tensor_parallel_size": 1,
    },
    tensor_parallel_size=4,
)
</syntaxhighlight>

=== CLI Configuration ===
<syntaxhighlight lang="bash">
# N-gram via CLI
vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --speculative-model [ngram] \
    --num-speculative-tokens 5 \
    --ngram-prompt-lookup-max 4

# EAGLE via CLI
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --speculative-model yuhuili/EAGLE-LLaMA3-Instruct-8B \
    --num-speculative-tokens 5
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Speculative_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
