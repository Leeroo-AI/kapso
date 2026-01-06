{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|Task Arithmetic|https://arxiv.org/abs/2212.04089]]
|-
! Domains
| [[domain::Model_Merging]], [[domain::Evaluation]], [[domain::Multi_Task]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Principle for evaluating merged adapter performance to verify capability retention from source adapters.

=== Description ===

Merge Evaluation assesses the quality of a merged adapter by testing on tasks from each source adapter. Merging can cause degradation on individual tasks (negative transfer), so evaluation helps determine if the merge was successful and whether to adjust parameters.

Key concerns:
* Does the merged adapter retain capabilities from each source?
* Is performance comparable to individual adapters?
* Are there signs of negative transfer or catastrophic forgetting?

=== Usage ===

Apply this principle after executing adapter merge:
* Test on representative samples from each source task
* Compare metrics to individual adapter baselines
* Iterate on merge parameters (weights, density) if needed

== Theoretical Basis ==

'''Evaluation Metrics:'''

For merged adapters, evaluate:
* Task-specific metrics (accuracy, perplexity, F1, etc.)
* Average performance across all source tasks
* Performance drop relative to individual adapters

'''Pareto Efficiency:'''

Ideal merges approach the Pareto frontier where improving one task would harm another. Significant degradation indicates the merge parameters need adjustment.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_peft_merged_adapter_evaluation]]
