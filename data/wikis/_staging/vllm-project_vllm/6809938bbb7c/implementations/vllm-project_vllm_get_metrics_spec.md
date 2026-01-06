{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::Monitoring]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

Concrete tool for retrieving speculative decoding metrics including draft counts, acceptance rates, and per-position statistics.

=== Description ===

`LLM.get_metrics()` returns Prometheus-style metrics including speculation-specific counters:
- **spec_decode_num_drafts:** Total speculation attempts
- **spec_decode_num_accepted_tokens:** Accepted speculated tokens
- **spec_decode_num_accepted_tokens_per_pos:** Acceptance by position

These metrics enable tuning speculation parameters for optimal performance.

=== Usage ===

Retrieve speculation metrics when:
- Tuning speculation depth (num_speculative_tokens)
- Comparing speculation methods
- Debugging poor acceptance rates
- Monitoring production performance

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project/vllm]
* '''File:''' vllm/entrypoints/llm.py
* '''Lines:''' L1539-1549

=== Signature ===
<syntaxhighlight lang="python">
def get_metrics(self) -> list[Metric]:
    """
    Get engine metrics including speculation statistics.

    Returns:
        List of Metric objects with counters and gauges.

    Speculation-specific metrics:
        - vllm:spec_decode_num_drafts (Counter)
        - vllm:spec_decode_num_accepted_tokens (Counter)
        - vllm:spec_decode_num_accepted_tokens_per_pos (Vector)
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm import LLM
# Use llm.get_metrics()
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| (none) || - || - || Method takes no parameters
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| metrics || list[Metric] || Prometheus-style metrics list
|-
| spec_decode_num_drafts || Counter || Total speculation attempts
|-
| spec_decode_num_accepted_tokens || Counter || Total accepted tokens
|}

== Usage Examples ==

=== Basic Metrics Retrieval ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    speculative_config={"method": "ngram", "num_speculative_tokens": 5},
)

# Generate some text
outputs = llm.generate(
    ["Explain quantum computing:"] * 10,
    SamplingParams(max_tokens=100),
)

# Get metrics
metrics = llm.get_metrics()

# Find speculation metrics
for metric in metrics:
    if "spec_decode" in metric.name:
        print(f"{metric.name}: {metric.value}")
</syntaxhighlight>

=== Calculate Acceptance Rate ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    speculative_config={"method": "ngram", "num_speculative_tokens": 5},
)

# Run workload
outputs = llm.generate(
    ["Write a story about a robot:"] * 20,
    SamplingParams(max_tokens=150),
)

# Get metrics
metrics = llm.get_metrics()

# Extract speculation stats
drafts = None
accepted = None
for metric in metrics:
    if metric.name == "vllm:spec_decode_num_drafts":
        drafts = metric.value
    elif metric.name == "vllm:spec_decode_num_accepted_tokens":
        accepted = metric.value

if drafts and accepted:
    # Each draft proposes num_speculative_tokens
    proposed = drafts * 5  # num_speculative_tokens
    acceptance_rate = accepted / proposed * 100
    print(f"Drafts: {drafts}")
    print(f"Accepted: {accepted}/{proposed}")
    print(f"Acceptance rate: {acceptance_rate:.1f}%")
</syntaxhighlight>

=== Per-Position Analysis ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    speculative_config={"method": "ngram", "num_speculative_tokens": 5},
)

# Generate
outputs = llm.generate(
    ["Write code:"] * 50,
    SamplingParams(max_tokens=100),
)

# Get per-position acceptance (if available)
metrics = llm.get_metrics()

for metric in metrics:
    if "per_pos" in metric.name.lower():
        print(f"Per-position acceptance: {metric.value}")
        # Shows acceptance rate drops with position
        # Position 1: ~80%, Position 5: ~50% (typical)
</syntaxhighlight>

=== Tuning Based on Metrics ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

def evaluate_speculation_config(model, spec_config, prompts):
    """Evaluate speculation configuration."""
    llm = LLM(model=model, speculative_config=spec_config)

    import time
    start = time.time()
    outputs = llm.generate(prompts, SamplingParams(max_tokens=100))
    elapsed = time.time() - start

    metrics = llm.get_metrics()

    result = {
        "time": elapsed,
        "tokens_per_sec": sum(len(o.outputs[0].token_ids) for o in outputs) / elapsed,
    }

    for m in metrics:
        if "spec_decode" in m.name:
            result[m.name] = m.value

    return result

# Compare configurations
prompts = ["Explain AI:"] * 20

configs = [
    {"method": "ngram", "num_speculative_tokens": 3},
    {"method": "ngram", "num_speculative_tokens": 5},
    {"method": "ngram", "num_speculative_tokens": 7},
]

for config in configs:
    result = evaluate_speculation_config(
        "meta-llama/Llama-3.2-1B-Instruct",
        config,
        prompts,
    )
    print(f"K={config['num_speculative_tokens']}: {result['tokens_per_sec']:.1f} tok/s")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Speculation_Metrics]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
