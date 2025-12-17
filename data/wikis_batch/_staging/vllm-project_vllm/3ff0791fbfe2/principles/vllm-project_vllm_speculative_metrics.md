= Speculative Metrics Analysis Principle =

== Metadata ==
{| class="wikitable"
|-
! Field !! Value
|-
| Knowledge Sources || vLLM Metrics Documentation, Prometheus Documentation, Speculative Decoding Papers
|-
| Domains || Performance Monitoring, Metrics Analysis, Optimization
|-
| Last Updated || 2025-12-17
|}

== Overview ==

The Speculative Metrics Analysis principle defines how to measure, monitor, and analyze the effectiveness of speculative decoding. It establishes metrics that quantify speculation performance, guide optimization decisions, and validate the cost-benefit tradeoffs of different speculative methods.

== Description ==

Effective speculative decoding requires continuous monitoring of acceptance rates, throughput, and resource utilization. This principle provides the framework for understanding speculation effectiveness and making data-driven optimization decisions.

=== Core Metrics ===

==== Acceptance Metrics ====

* '''num_drafts''': Total number of speculation attempts
* '''num_draft_tokens''': Total tokens proposed by draft method
* '''num_accepted_tokens''': Total tokens accepted after verification
* '''num_accepted_tokens_per_pos''': Acceptance count for each speculative position (0 to k-1)

==== Derived Metrics ====

* '''Draft Acceptance Rate''': <code>num_accepted_tokens / num_draft_tokens</code>
  - Percentage of drafted tokens that were accepted
  - Higher is better (>60% is good, >80% is excellent)
  - Indicates draft quality

* '''Mean Acceptance Length''': <code>1 + (num_accepted_tokens / num_drafts)</code>
  - Average number of tokens accepted per speculation round (including bonus token)
  - Direct indicator of speedup (2.0 means ~2x speedup)
  - Ranges from 1.0 (no speedup) to 1+k (perfect speculation)

* '''Per-Position Acceptance Rate''': <code>num_accepted_tokens_per_pos[i] / num_drafts</code>
  - Acceptance rate for each position in the speculative window
  - Shows how far speculation typically succeeds
  - Helps tune num_speculative_tokens

==== Throughput Metrics ====

* '''Draft Throughput''': Tokens proposed per second
* '''Accepted Throughput''': Accepted tokens per second
* '''Effective Speedup''': Ratio of speculative to non-speculative throughput

=== Metric Interpretation ===

==== Healthy Speculation ====
* Mean acceptance length: 1.5-2.5+
* Draft acceptance rate: 60-90%
* Position 0 acceptance: >90%
* Position k-1 acceptance: >20%

==== Poor Speculation ====
* Mean acceptance length: <1.3
* Draft acceptance rate: <40%
* Position 0 acceptance: <70%
* Rapid decline in per-position acceptance

==== Optimal Speculation ====
* Mean acceptance length: >2.0
* Draft acceptance rate: >70%
* Smooth decline in per-position acceptance
* Consistent performance across requests

=== Analysis Strategies ===

==== Method Comparison ====
Compare metrics across speculative methods to choose best for workload:
* EAGLE typically has highest acceptance length (2.0-2.8)
* Ngram effective for repetitive content (1.5-2.0)
* MLP speculator balanced (1.5-2.0)
* Suffix excels at specific patterns (2.0-3.0+)

==== Parameter Tuning ====
Use metrics to optimize num_speculative_tokens:
* If position k-1 acceptance >30%: increase k
* If position k-1 acceptance <10%: decrease k
* Monitor mean acceptance length as k changes
* Balance acceptance rate vs overhead

==== Workload Characterization ====
Analyze metrics across different tasks:
* Code generation: High ngram acceptance
* Creative writing: Better with EAGLE
* Question answering: Mixed results
* Repetitive tasks: Excellent for suffix

=== Monitoring Strategy ===

==== Real-Time Monitoring ====
* Aggregate metrics over time windows (30s, 5min)
* Alert on sudden acceptance rate drops
* Track trends in mean acceptance length
* Monitor resource utilization

==== Per-Request Analysis ====
* Track acceptance rate per request type
* Identify requests that benefit most
* Detect pathological cases
* Guide request routing

==== A/B Testing ====
* Compare speculation on/off for same workload
* Test different methods side-by-side
* Validate speedup claims
* Measure end-to-end latency impact

== Usage Context ==

This principle applies when:

* Deploying speculative decoding in production
* Tuning speculation parameters
* Comparing speculative methods
* Debugging performance issues
* Validating optimization efforts

Metrics should be:
* Collected continuously during inference
* Aggregated at appropriate time scales
* Analyzed for trends and anomalies
* Used to guide configuration changes

== Design Considerations ==

=== Trade-offs ===

* '''Granularity vs. Overhead''': Detailed per-request metrics add overhead
* '''Aggregation Window''': Short windows show variance, long windows hide issues
* '''Storage Cost''': Long-term metric retention requires storage
* '''Privacy''': Per-request metrics may contain sensitive information

=== Implementation Strategies ===

==== Prometheus Integration ====
* Export metrics in Prometheus format
* Use counters for cumulative values
* Support PromQL queries for analysis
* Enable Grafana dashboards

==== Snapshot API ====
* Provide get_metrics() for programmatic access
* Return current counter values
* Support metric filtering
* Enable custom analysis tools

==== Logging Integration ====
* Periodic log output with aggregated metrics
* Structured log format for parsing
* Configurable log intervals
* Integration with log aggregation systems

=== Metric Collection Patterns ===

==== Aggregation ====
* Sum across all requests in time window
* Calculate rates using time deltas
* Compute percentiles for distributions
* Track min/max/mean values

==== Filtering ====
* Per-model metrics (multi-model serving)
* Per-method metrics (comparing methods)
* Per-request-type metrics (different workloads)
* Per-user metrics (multi-tenant systems)

==== Alerting ====
* Acceptance rate below threshold
* Speedup regression detected
* Resource utilization anomalies
* Error rate increases

== Performance Implications ==

=== Overhead Analysis ===

* '''Metric Collection''': <1% overhead (atomic counter updates)
* '''Prometheus Export''': Negligible (scrape-based)
* '''Per-Position Tracking''': O(k) memory per metric snapshot
* '''API Calls''': get_metrics() is fast (counter reads)

=== Optimization Decisions ===

Use metrics to:
* '''Increase k''': If position k-1 acceptance >30%
* '''Decrease k''': If mean acceptance length drops
* '''Disable Speculation''': If acceptance length <1.2 consistently
* '''Switch Method''': If another method shows better metrics
* '''Adjust Batch Size''': Based on speculation effectiveness

=== Cost-Benefit Analysis ===

Calculate ROI of speculation:
<syntaxhighlight lang="text">
Speedup = mean_acceptance_length
Cost = draft_model_memory + computation_overhead
Benefit = (speedup - 1) * baseline_throughput
ROI = benefit / cost
</syntaxhighlight>

== Metric Use Cases ==

=== Use Case 1: Method Selection ===
Run same workload with different methods, compare mean acceptance length:
* EAGLE: 2.3 → Best for this workload
* Ngram: 1.6 → Acceptable fallback
* MLP: 1.8 → Balanced choice

=== Use Case 2: Parameter Tuning ===
Test different num_speculative_tokens:
* k=3: mean=1.8, pos[2]=0.35 → Try k=4
* k=4: mean=2.1, pos[3]=0.28 → Good balance
* k=5: mean=2.0, pos[4]=0.15 → Too aggressive

=== Use Case 3: Workload Optimization ===
Analyze per-request-type:
* Code generation: 85% acceptance → Keep speculation
* Creative writing: 45% acceptance → Consider disabling
* QA: 70% acceptance → Good performance

=== Use Case 4: Production Monitoring ===
Continuous monitoring shows:
* Normal: mean acceptance length = 2.1 ± 0.2
* Alert: Drops to 1.4 → Investigate
* Root cause: Model drift or workload change
* Action: Retune or switch method

== Related Principles ==

* [[implements::vllm-project_vllm_get_metrics]] - Metrics retrieval implementation
* Performance Monitoring Principles - System-wide monitoring
* Optimization Principles - Using metrics to guide optimization
* A/B Testing Principles - Comparative evaluation

== See Also ==

* [[implemented_by::Implementation:vllm-project_vllm_get_metrics]]
* [[implements::vllm-project_vllm_LLM_generate_spec]]
* Prometheus Metrics Documentation
* vLLM Observability Guide
