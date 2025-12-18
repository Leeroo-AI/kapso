# Principle: Priority Assignment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Similarity_Metrics]], [[domain::Workflow_Evaluation]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Principle for categorizing edit operations by severity based on cost thresholds, node type, and operation characteristics.

=== Description ===

Priority Assignment classifies edit operations into actionable severity levels:

1. **Cost-Based Thresholds**: Compare operation cost against configuration thresholds
2. **Node Type Consideration**: Trigger nodes receive elevated priority
3. **Operation Type Impact**: Insertions and deletions weighted differently than substitutions
4. **Three-Level Classification**: Critical, Major, Minor

Classification rules:
- **Critical**: Trigger node insertions/deletions OR cost ≥ 80% of trigger substitution threshold
- **Major**: Cost ≥ 80% of different-type substitution threshold
- **Minor**: All other operations below major threshold

=== Usage ===

Apply this principle when:
- Building actionable reports from similarity metrics
- Filtering edits to focus on high-impact differences
- Creating pass/fail criteria based on severity counts
- Guiding users to fix most important workflow differences first

== Theoretical Basis ==

Priority assignment uses a tiered threshold approach:

<syntaxhighlight lang="python">
# Priority determination logic

def determine_priority(cost, config, node_data=None, operation_type=None):
    # Rule 1: Trigger operations are critical (except minor updates)
    if node_data and node_data.get("is_trigger", False):
        if operation_type in ("node_insert", "node_delete"):
            return "critical"

    # Rule 2: Cost-based thresholds with 0.8 safety margin
    trigger_threshold = config.node_substitution_trigger * 0.8
    type_change_threshold = config.node_substitution_different_type * 0.8

    if cost >= trigger_threshold:
        return "critical"
    elif cost >= type_change_threshold:
        return "major"
    else:
        return "minor"
</syntaxhighlight>

The 0.8 multiplier provides a safety margin, catching operations that approach (but don't quite reach) the threshold costs.

Example with default config:
- node_substitution_trigger = 10.0 → critical threshold = 8.0
- node_substitution_different_type = 8.0 → major threshold = 6.4
- Operation with cost 7.0 → "major" (≥6.4 but <8.0)

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_determine_priority]]
