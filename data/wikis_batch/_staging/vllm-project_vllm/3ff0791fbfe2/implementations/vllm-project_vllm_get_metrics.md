= LLM.get_metrics() for Speculative Decoding =

== Metadata ==
{| class="wikitable"
|-
! Field !! Value
|-
| Knowledge Sources || vLLM Metrics API, Prometheus Documentation, examples/offline_inference/spec_decode.py
|-
| Domains || Metrics Collection, Performance Analysis, Monitoring
|-
| Last Updated || 2025-12-17
|}

== Overview ==

The <code>LLM.get_metrics()</code> method retrieves a snapshot of aggregated metrics from the vLLM engine, including detailed speculative decoding statistics. This enables programmatic analysis of speculation effectiveness and performance optimization.

== Code Reference ==

=== Source Location ===
<syntaxhighlight lang="text">
File: vllm/entrypoints/llm.py
Class: LLM
Method: get_metrics()
Lines: 1539-1549

Metrics Implementation:
File: vllm/v1/spec_decode/metrics.py
Classes: SpecDecodingStats, SpecDecodingProm
Lines: 1-226
</syntaxhighlight>

=== Signature ===
<syntaxhighlight lang="python">
def get_metrics(self) -> list["Metric"]:
    """Return a snapshot of aggregated metrics from Prometheus.

    Returns:
        A list of Metric instances capturing the current state
        of all aggregated metrics from Prometheus.

    Note:
        This method is only available with the V1 LLM engine.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Metric, Counter, Vector
</syntaxhighlight>

== Description ==

<code>get_metrics()</code> implements the [[implements::Principle:vllm-project_vllm_speculative_metrics]] principle by providing programmatic access to speculative decoding performance metrics. It returns a list of <code>Metric</code> objects that can be analyzed to understand speculation effectiveness.

=== Metric Types ===

* '''Counter''': Cumulative count metric (e.g., total accepted tokens)
* '''Vector''': List of values indexed by position (e.g., per-position acceptance)

=== Available Metrics ===

When speculative decoding is enabled, the following metrics are available:

{| class="wikitable"
! Metric Name !! Type !! Description
|-
| vllm:spec_decode_num_drafts || Counter || Total number of draft speculation rounds
|-
| vllm:spec_decode_num_draft_tokens || Counter || Total tokens proposed by draft method
|-
| vllm:spec_decode_num_accepted_tokens || Counter || Total tokens accepted after verification
|-
| vllm:spec_decode_num_accepted_tokens_per_pos || Vector || Acceptance count for each position [0, k-1]
|}

== Input/Output Contract ==

=== Input ===

No parameters required - method is called on LLM instance:
<syntaxhighlight lang="python">
metrics = llm.get_metrics()
</syntaxhighlight>

=== Output ===

Returns <code>list[Metric]</code> where each metric has:

{| class="wikitable"
! Attribute !! Type !! Description
|-
| name || str || Metric identifier (e.g., "vllm:spec_decode_num_drafts")
|-
| value || float/int || Current value (for Counter)
|-
| values || list[float] || List of values (for Vector)
|}

== Usage Examples ==

=== Example 1: Basic Metrics Retrieval ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

# Initialize with speculation
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
    }
)

# Generate some text
prompts = ["Write a story about"] * 10
outputs = llm.generate(prompts, SamplingParams(max_tokens=100))

# Retrieve metrics
metrics = llm.get_metrics()

# Display all spec decode metrics
for metric in metrics:
    if "spec_decode" in metric.name:
        if isinstance(metric, Counter):
            print(f"{metric.name}: {metric.value}")
        elif isinstance(metric, Vector):
            print(f"{metric.name}: {metric.values}")
</syntaxhighlight>

=== Example 2: Calculate Acceptance Rate ===
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

# Generate text
outputs = llm.generate(
    ["Explain quantum computing"] * 20,
    SamplingParams(temperature=0.0, max_tokens=150)
)

# Get metrics
metrics = llm.get_metrics()

# Extract key values
num_drafts = 0
num_draft_tokens = 0
num_accepted_tokens = 0

for metric in metrics:
    if metric.name == "vllm:spec_decode_num_drafts":
        num_drafts = metric.value
    elif metric.name == "vllm:spec_decode_num_draft_tokens":
        num_draft_tokens = metric.value
    elif metric.name == "vllm:spec_decode_num_accepted_tokens":
        num_accepted_tokens = metric.value

# Calculate acceptance metrics
if num_draft_tokens > 0:
    acceptance_rate = (num_accepted_tokens / num_draft_tokens) * 100
    print(f"Draft acceptance rate: {acceptance_rate:.1f}%")

if num_drafts > 0:
    mean_acceptance_length = 1 + (num_accepted_tokens / num_drafts)
    print(f"Mean acceptance length: {mean_acceptance_length:.2f}")
    print(f"Estimated speedup: {mean_acceptance_length:.2f}x")
</syntaxhighlight>

=== Example 3: Per-Position Analysis ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Vector

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "eagle",
        "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        "num_speculative_tokens": 5,
    }
)

# Generate
outputs = llm.generate(
    ["Write about AI"] * 50,
    SamplingParams(temperature=0.0, max_tokens=200)
)

# Get metrics
metrics = llm.get_metrics()

# Find per-position acceptance
num_drafts = 0
acceptance_per_pos = None

for metric in metrics:
    if metric.name == "vllm:spec_decode_num_drafts":
        num_drafts = metric.value
    elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
        assert isinstance(metric, Vector)
        acceptance_per_pos = metric.values

# Calculate per-position acceptance rates
if num_drafts > 0 and acceptance_per_pos:
    print("Per-position acceptance rates:")
    for i, count in enumerate(acceptance_per_pos):
        rate = (count / num_drafts) * 100
        print(f"  Position {i}: {rate:.1f}% ({int(count)}/{int(num_drafts)})")

    # Determine if we should increase num_speculative_tokens
    last_pos_rate = (acceptance_per_pos[-1] / num_drafts) * 100
    if last_pos_rate > 30:
        print("\nSuggestion: Consider increasing num_speculative_tokens")
    elif last_pos_rate < 10:
        print("\nSuggestion: Consider decreasing num_speculative_tokens")
</syntaxhighlight>

=== Example 4: Comparative Analysis ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

def benchmark_method(method_config, prompts):
    """Benchmark a speculative method and return metrics."""
    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        speculative_config=method_config,
    )

    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=100)
    )

    metrics = llm.get_metrics()

    # Extract key metrics
    num_drafts = 0
    num_accepted = 0

    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            num_drafts = metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            num_accepted = metric.value

    mean_acceptance = 1 + (num_accepted / num_drafts) if num_drafts > 0 else 1.0

    return {
        "method": method_config["method"],
        "mean_acceptance_length": mean_acceptance,
        "num_outputs": len(outputs),
    }

# Test prompts
prompts = ["Explain machine learning"] * 30

# Compare methods
methods = [
    {"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4},
    {"method": "eagle", "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
     "num_speculative_tokens": 3},
]

results = []
for method_config in methods:
    result = benchmark_method(method_config, prompts)
    results.append(result)
    print(f"{result['method']}: {result['mean_acceptance_length']:.2f}x speedup")

# Choose best method
best = max(results, key=lambda r: r["mean_acceptance_length"])
print(f"\nBest method: {best['method']} ({best['mean_acceptance_length']:.2f}x)")
</syntaxhighlight>

=== Example 5: Production Monitoring ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
import time

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
    }
)

def monitor_metrics(llm, interval_seconds=60):
    """Monitor metrics at regular intervals."""
    prev_metrics = {}

    while True:
        time.sleep(interval_seconds)

        metrics = llm.get_metrics()

        # Calculate delta metrics
        current_metrics = {}
        for metric in metrics:
            if "spec_decode" in metric.name:
                if hasattr(metric, "value"):
                    current_metrics[metric.name] = metric.value

        # Compute interval metrics
        if prev_metrics:
            drafts_delta = current_metrics.get("vllm:spec_decode_num_drafts", 0) - \
                          prev_metrics.get("vllm:spec_decode_num_drafts", 0)
            accepted_delta = current_metrics.get("vllm:spec_decode_num_accepted_tokens", 0) - \
                           prev_metrics.get("vllm:spec_decode_num_accepted_tokens", 0)

            if drafts_delta > 0:
                interval_acceptance = 1 + (accepted_delta / drafts_delta)
                print(f"[{time.strftime('%H:%M:%S')}] "
                      f"Acceptance length: {interval_acceptance:.2f}, "
                      f"Drafts: {drafts_delta}")

                # Alert on degradation
                if interval_acceptance < 1.3:
                    print("WARNING: Low acceptance rate detected!")

        prev_metrics = current_metrics

# In production, run monitoring in background thread
# monitor_metrics(llm, interval_seconds=60)
</syntaxhighlight>

=== Example 6: Complete Analysis from spec_decode.py ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.v1.metrics.reader import Counter, Vector
from transformers import AutoTokenizer

# Initialize
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "eagle",
        "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        "num_speculative_tokens": 3,
    }
)

# Generate
prompts = ["Explain AI"] * 80
prompt_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
tokens_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids]

outputs = llm.generate(
    tokens_prompts,
    SamplingParams(temperature=0.0, max_tokens=256)
)

# Comprehensive metrics analysis
metrics = llm.get_metrics()

total_num_output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
num_drafts = 0
num_draft_tokens = 0
num_accepted_tokens = 0
acceptance_counts = [0] * 3  # num_speculative_tokens

for metric in metrics:
    if metric.name == "vllm:spec_decode_num_drafts":
        assert isinstance(metric, Counter)
        num_drafts += metric.value
    elif metric.name == "vllm:spec_decode_num_draft_tokens":
        assert isinstance(metric, Counter)
        num_draft_tokens += metric.value
    elif metric.name == "vllm:spec_decode_num_accepted_tokens":
        assert isinstance(metric, Counter)
        num_accepted_tokens += metric.value
    elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
        assert isinstance(metric, Vector)
        for pos in range(len(metric.values)):
            acceptance_counts[pos] += metric.values[pos]

print("-" * 50)
print(f"total_num_output_tokens: {total_num_output_tokens}")
print(f"num_drafts: {num_drafts}")
print(f"num_draft_tokens: {num_draft_tokens}")
print(f"num_accepted_tokens: {num_accepted_tokens}")

acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1
print(f"mean acceptance length: {acceptance_length:.2f}")
print("-" * 50)

# Per-position acceptance rates
for i in range(len(acceptance_counts)):
    acceptance_rate = acceptance_counts[i] / num_drafts if num_drafts > 0 else 0
    print(f"acceptance at token {i}: {acceptance_rate:.2f}")
</syntaxhighlight>

== Design Details ==

=== Metric Storage ===

Metrics are stored as Prometheus counters:
* Atomic updates during inference
* Cumulative values (never decrease)
* Thread-safe access
* Minimal overhead (<1%)

=== Snapshot Behavior ===

<code>get_metrics()</code> returns current values at call time:
* Not a reset operation (counters persist)
* Values continue to accumulate
* Calculate deltas by calling multiple times
* No locking required (atomic reads)

=== Metric Scope ===

* Per-engine metrics (multiple LLM instances maintain separate metrics)
* Aggregated across all requests
* No per-request breakdown in snapshot
* Use logging for detailed per-request analysis

=== Performance ===

* '''Call Cost''': O(num_metrics) - very fast
* '''Memory''': Minimal (small number of counters)
* '''Thread Safety''': Fully thread-safe
* '''Overhead''': Negligible on inference performance

== Analysis Patterns ==

=== Pattern: Delta Calculation ===
<syntaxhighlight lang="python">
# First snapshot
metrics1 = llm.get_metrics()
extract_counters(metrics1)  # Helper to extract values

# Generate more
llm.generate(prompts, sampling_params)

# Second snapshot
metrics2 = llm.get_metrics()
extract_counters(metrics2)

# Calculate delta
delta_drafts = counter2_drafts - counter1_drafts
delta_accepted = counter2_accepted - counter1_accepted
</syntaxhighlight>

=== Pattern: Acceptance Rate Monitoring ===
<syntaxhighlight lang="python">
def get_acceptance_rate(metrics):
    drafts = accepted = 0
    for m in metrics:
        if m.name == "vllm:spec_decode_num_drafts":
            drafts = m.value
        elif m.name == "vllm:spec_decode_num_accepted_tokens":
            accepted = m.value
    return 1 + (accepted / drafts) if drafts > 0 else 1.0

rate = get_acceptance_rate(llm.get_metrics())
print(f"Current acceptance length: {rate:.2f}x")
</syntaxhighlight>

=== Pattern: Position Analysis ===
<syntaxhighlight lang="python">
def analyze_positions(metrics, num_spec_tokens):
    drafts = 0
    per_pos = None

    for m in metrics:
        if m.name == "vllm:spec_decode_num_drafts":
            drafts = m.value
        elif m.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            per_pos = m.values

    if drafts > 0 and per_pos:
        rates = [count / drafts for count in per_pos]
        return rates
    return []

rates = analyze_positions(llm.get_metrics(), num_speculative_tokens=5)
print(f"Position acceptance rates: {rates}")
</syntaxhighlight>

== Common Use Cases ==

=== A/B Testing ===
Compare speculation vs. no speculation using metrics:
<syntaxhighlight lang="python">
# With speculation
llm_spec = LLM(model="...", speculative_config={...})
outputs_spec = llm_spec.generate(prompts, params)
metrics_spec = llm_spec.get_metrics()

# Without speculation
llm_base = LLM(model="...")
outputs_base = llm_base.generate(prompts, params)

# Compare using metrics_spec to validate speedup
</syntaxhighlight>

=== Parameter Tuning ===
Iteratively test num_speculative_tokens values:
<syntaxhighlight lang="python">
for k in [3, 4, 5, 6]:
    llm = LLM(model="...", speculative_config={"num_speculative_tokens": k, ...})
    outputs = llm.generate(prompts, params)
    metrics = llm.get_metrics()
    # Analyze and choose best k
</syntaxhighlight>

=== Production Dashboards ===
Export metrics to monitoring systems:
<syntaxhighlight lang="python">
metrics = llm.get_metrics()
# Push to Prometheus, Grafana, CloudWatch, etc.
# Visualize trends over time
# Set alerts on degradation
</syntaxhighlight>

== Related Pages ==

* [[implements::Principle:vllm-project_vllm_speculative_metrics]]
* [[implemented_by::vllm-project_vllm_LLM_generate_spec]]
* Prometheus Metrics Documentation
* SpecDecodingStats Class
* SpecDecodingProm Class
