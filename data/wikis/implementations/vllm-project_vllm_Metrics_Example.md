{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::Metrics]], [[domain::Performance Monitoring]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
This example demonstrates how to access and display vLLM performance metrics for monitoring inference workloads.

=== Description ===
The metrics example shows how to retrieve and interpret performance metrics from vLLM's v1 engine using the get_metrics() method. The example demonstrates accessing four types of metrics: Gauge (current values), Counter (cumulative counts), Vector (multi-dimensional values), and Histogram (distribution data with buckets). These metrics provide insights into throughput, latency, GPU utilization, and other performance characteristics. This is essential for production monitoring, performance tuning, and identifying bottlenecks.

=== Usage ===
Use this pattern when you need to monitor vLLM performance in production, debug performance issues, collect metrics for dashboards or alerting systems, or understand system behavior under different workloads. Enable metrics by setting disable_log_stats=False when creating the LLM.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/metrics.py examples/offline_inference/metrics.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
python examples/offline_inference/metrics.py
</syntaxhighlight>

== Key Concepts ==

=== Metric Types ===
vLLM provides four metric types from vllm.v1.metrics.reader:

'''Gauge''': Current instantaneous value
* Example: Current GPU memory usage, active requests
* Access via: metric.value

'''Counter''': Cumulative count that only increases
* Example: Total requests processed, total tokens generated
* Access via: metric.value

'''Vector''': Multi-dimensional metric with labeled values
* Example: Per-GPU memory usage, per-layer latency
* Access via: metric.values (dict)

'''Histogram''': Distribution of values with buckets
* Example: Request latency distribution, token counts
* Access via: metric.sum, metric.count, metric.buckets

=== Metrics Collection ===
Metrics are accessed via the LLM.get_metrics() method:
* Returns list of all available metrics
* Collected after inference completes
* Requires disable_log_stats=False during LLM initialization
* Each metric has a name and type-specific data

=== Common Metrics ===
Typical metrics include:
* Throughput: Tokens per second, requests per second
* Latency: Time to first token (TTFT), inter-token latency
* GPU utilization: Memory usage, compute utilization
* Queue depth: Pending requests, running requests
* Cache statistics: KV cache hit rate, cache memory

== Usage Examples ==

=== Enabling and Retrieving Metrics ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Gauge, Histogram, Vector

# Enable metrics collection
llm = LLM(
    model="facebook/opt-125m",
    disable_log_stats=False  # Enable metrics
)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Run inference
outputs = llm.generate(prompts, sampling_params)

# Print outputs
for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
    print("-" * 50)
</syntaxhighlight>

=== Processing Different Metric Types ===
<syntaxhighlight lang="python">
# Retrieve and display all metrics
for metric in llm.get_metrics():
    if isinstance(metric, Gauge):
        print(f"{metric.name} (gauge) = {metric.value}")

    elif isinstance(metric, Counter):
        print(f"{metric.name} (counter) = {metric.value}")

    elif isinstance(metric, Vector):
        print(f"{metric.name} (vector) = {metric.values}")

    elif isinstance(metric, Histogram):
        print(f"{metric.name} (histogram)")
        print(f"    sum = {metric.sum}")
        print(f"    count = {metric.count}")
        for bucket_le, value in metric.buckets.items():
            print(f"    {bucket_le} = {value}")
</syntaxhighlight>

=== Filtering Specific Metrics ===
<syntaxhighlight lang="python">
# Get only throughput metrics
throughput_metrics = [
    m for m in llm.get_metrics()
    if "throughput" in m.name.lower()
]

for metric in throughput_metrics:
    print(f"{metric.name}: {metric.value}")

# Get latency histogram
latency_histograms = [
    m for m in llm.get_metrics()
    if isinstance(m, Histogram) and "latency" in m.name.lower()
]

for hist in latency_histograms:
    print(f"{hist.name}:")
    print(f"  Average: {hist.sum / hist.count if hist.count > 0 else 0:.2f}")
    print(f"  Count: {hist.count}")
</syntaxhighlight>

=== Integration with Monitoring Systems ===
<syntaxhighlight lang="python">
import json

def export_metrics_as_json(llm):
    """Export metrics in JSON format for monitoring systems."""
    metrics_data = {}

    for metric in llm.get_metrics():
        if isinstance(metric, (Gauge, Counter)):
            metrics_data[metric.name] = {
                "type": type(metric).__name__.lower(),
                "value": metric.value
            }
        elif isinstance(metric, Vector):
            metrics_data[metric.name] = {
                "type": "vector",
                "values": metric.values
            }
        elif isinstance(metric, Histogram):
            metrics_data[metric.name] = {
                "type": "histogram",
                "sum": metric.sum,
                "count": metric.count,
                "buckets": metric.buckets
            }

    return json.dumps(metrics_data, indent=2)

# Export to JSON
metrics_json = export_metrics_as_json(llm)
print(metrics_json)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
