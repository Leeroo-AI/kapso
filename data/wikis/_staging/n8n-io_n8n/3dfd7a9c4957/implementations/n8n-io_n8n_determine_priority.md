# Implementation: _determine_priority

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

Concrete function for classifying edit operations into priority levels based on cost, node type, and operation characteristics.

=== Description ===

`_determine_priority()` applies a multi-rule classification:

1. **Trigger Rule**: Trigger node insertions/deletions are always critical
2. **Cost Thresholds**: Compare against scaled config thresholds (×0.8)
3. **Return Value**: One of `'critical'`, `'major'`, or `'minor'`

=== Usage ===

This function is called internally by `_extract_operations_from_path()` when processing each edit operation. It's not typically called directly.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/similarity.py
* '''Lines:''' L389-418

=== Signature ===
<syntaxhighlight lang="python">
def _determine_priority(
    cost: float,
    config: WorkflowComparisonConfig,
    node_data: Optional[Dict[str, Any]] = None,
    operation_type: Optional[str] = None,
) -> str:
    """
    Determine priority level based on cost, node type, and operation.

    Args:
        cost: Edit operation cost
        config: Configuration
        node_data: Optional node data to check if it's a trigger
        operation_type: Type of operation (node_insert, node_delete, node_substitute, etc.)

    Returns:
        Priority level: 'critical', 'major', or 'minor'
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Internal function - not typically imported directly
from src.similarity import _determine_priority
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| cost || float || Yes || Edit operation cost value
|-
| config || WorkflowComparisonConfig || Yes || Configuration with threshold values
|-
| node_data || Optional[Dict[str, Any]] || No || Node data dict, checked for `is_trigger` key
|-
| operation_type || Optional[str] || No || Operation type: `node_insert`, `node_delete`, `node_substitute`, etc.
|}

=== Outputs ===
{| class="wikitable"
|-
! Return Type !! Description
|-
| str || Priority level: `'critical'`, `'major'`, or `'minor'`
|}

=== Classification Rules ===
{| class="wikitable"
|-
! Rule !! Condition !! Result
|-
| Trigger Operations || `is_trigger=True` AND `operation_type in (node_insert, node_delete)` || `'critical'`
|-
| High Cost || `cost >= config.node_substitution_trigger * 0.8` || `'critical'`
|-
| Medium Cost || `cost >= config.node_substitution_different_type * 0.8` || `'major'`
|-
| Low Cost || Otherwise || `'minor'`
|}

== Usage Examples ==

=== Implementation ===
<syntaxhighlight lang="python">
def _determine_priority(
    cost: float,
    config: WorkflowComparisonConfig,
    node_data: Optional[Dict[str, Any]] = None,
    operation_type: Optional[str] = None,
) -> str:
    # Critical: trigger deletions/insertions (but not minor parameter updates)
    if node_data and node_data.get("is_trigger", False):
        if operation_type in ("node_insert", "node_delete"):
            return "critical"

    # Critical: trigger mismatches and high-cost operations
    if cost >= config.node_substitution_trigger * 0.8:
        return "critical"
    elif cost >= config.node_substitution_different_type * 0.8:
        return "major"
    else:
        return "minor"
</syntaxhighlight>

=== Example Classifications ===
<syntaxhighlight lang="python">
# Example 1: Trigger node deletion → always critical
priority = _determine_priority(
    cost=5.0,
    config=config,
    node_data={"type": "n8n-nodes-base.webhook", "is_trigger": True},
    operation_type="node_delete"
)
# Result: "critical" (trigger rule)

# Example 2: High-cost operation → critical
priority = _determine_priority(
    cost=9.0,  # >= 10.0 * 0.8 = 8.0
    config=config,  # node_substitution_trigger = 10.0
    node_data=None,
    operation_type="node_substitute"
)
# Result: "critical" (cost threshold)

# Example 3: Medium-cost operation → major
priority = _determine_priority(
    cost=7.0,  # >= 8.0 * 0.8 = 6.4, but < 8.0
    config=config,  # node_substitution_different_type = 8.0
    node_data=None,
    operation_type="node_substitute"
)
# Result: "major"

# Example 4: Low-cost operation → minor
priority = _determine_priority(
    cost=2.0,  # < 6.4
    config=config,
    node_data=None,
    operation_type="edge_insert"
)
# Result: "minor"
</syntaxhighlight>

=== Integration with Extract Operations ===
<syntaxhighlight lang="python">
# In _extract_operations_from_path()

for u, v in node_edit_path:
    if u is None:
        # Node insertion
        cost = config.node_insertion_cost
        operation = {
            "type": "node_insert",
            "cost": cost,
            "priority": _determine_priority(
                cost, config, g2.nodes[v], "node_insert"
            ),
            ...
        }
    elif v is None:
        # Node deletion
        cost = config.node_deletion_cost
        operation = {
            "type": "node_delete",
            "cost": cost,
            "priority": _determine_priority(
                cost, config, g1.nodes[u], "node_delete"
            ),
            ...
        }
    # ... etc
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Priority_Assignment]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Workflow_Comparison_Env]]
