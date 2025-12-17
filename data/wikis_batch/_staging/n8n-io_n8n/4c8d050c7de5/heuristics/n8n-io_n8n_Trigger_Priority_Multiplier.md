# Heuristic: Trigger Priority Multiplier

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n Workflow Comparison|https://github.com/n8n-io/n8n/tree/master/packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python]]
|-
! Domains
| [[domain::Graph_Algorithms]], [[domain::AI_Evaluation]], [[domain::Workflow_Automation]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

Trigger node substitution costs 50x the base substitution cost (default 50.0 vs 1.0), heavily penalizing incorrect trigger node types in workflow comparisons.

=== Description ===

In n8n workflow comparison, trigger nodes (nodes with no incoming edges) serve as workflow entry points and are semantically critical. Getting the trigger node wrong fundamentally changes the workflow's purpose. The Graph Edit Distance (GED) algorithm applies a 50x multiplier to trigger node substitution costs to reflect this criticality.

=== Usage ===

This heuristic is applied automatically by the **GED cost functions** when comparing workflows. Understanding this multiplier is critical when:

- Interpreting similarity scores where trigger nodes differ
- Creating custom comparison configurations
- Understanding why workflows with same structure but different triggers score low

== The Insight (Rule of Thumb) ==

* **Default Costs:**
  - `node_substitution_same_type = 1.0` (same node type)
  - `node_substitution_similar_type = 5.0` (similar group)
  - `node_substitution_different_type = 15.0` (different types)
  - `node_substitution_trigger = 50.0` (trigger mismatch - **50x base!**)
* **Insertion/Deletion:** `node_insertion_cost = 10.0`, `node_deletion_cost = 10.0`
* **Trade-off:** High trigger cost ensures workflow "intent" is weighted heavily; may over-penalize workflows that differ only in trigger mechanism

== Reasoning ==

The 50x multiplier for trigger nodes reflects domain-specific workflow semantics:

1. **Trigger = Intent:** A "Schedule Trigger" vs "Webhook Trigger" represents fundamentally different workflow purposes (time-based vs event-based)
2. **Entry Point Criticality:** Unlike middle nodes that can often be substituted, triggers define the workflow's activation contract
3. **AI Evaluation Context:** When evaluating AI-generated workflows, correct trigger selection is a primary success criterion
4. **Cost Hierarchy:**
   - Same type: 1.0 (minor parameter differences)
   - Similar type: 5.0 (functionally similar, e.g., different HTTP nodes)
   - Different type: 15.0 (different functionality)
   - Trigger mismatch: 50.0 (different workflow intent)

== Code Evidence ==

Default costs from `config_loader.py:103-109`:
<syntaxhighlight lang="python">
# Cost weights
node_insertion_cost: float = 10.0
node_deletion_cost: float = 10.0
node_substitution_same_type: float = 1.0
node_substitution_similar_type: float = 5.0
node_substitution_different_type: float = 15.0
node_substitution_trigger: float = 50.0  # 50x base cost!
</syntaxhighlight>

Config parsing from `config_loader.py:284`:
<syntaxhighlight lang="python">
config.node_substitution_trigger = subst.get("trigger_mismatch", 50.0)
</syntaxhighlight>

Config structure from `config_loader.py:233-234`:
<syntaxhighlight lang="python">
"substitution": {
    ...
    "trigger_mismatch": self.node_substitution_trigger,
},
</syntaxhighlight>

Trigger detection in graph building (from graph_builder.py, nodes with in_degree == 0 are triggers):
<syntaxhighlight lang="python">
# Trigger nodes are identified by having no incoming edges
# In the cost function, comparing a trigger to a non-trigger
# or comparing triggers of different types incurs the 50x penalty
</syntaxhighlight>

== Related Pages ==

* [[uses_heuristic::Implementation:n8n-io_n8n_calculate_graph_edit_distance]]
* [[uses_heuristic::Implementation:n8n-io_n8n_load_config]]
* [[uses_heuristic::Principle:n8n-io_n8n_GED_Calculation]]
* [[uses_heuristic::Workflow:n8n-io_n8n_Workflow_Comparison]]
