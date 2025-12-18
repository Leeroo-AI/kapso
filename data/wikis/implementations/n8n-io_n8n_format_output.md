# Implementation: format_output_json / format_output_summary

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::CLI_Tools]], [[domain::Workflow_Evaluation]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Concrete functions for transforming comparison results into JSON and human-readable summary formats.

=== Description ===

Two complementary formatting functions:

1. **`format_output_json()`**: Produces machine-readable JSON output
   - Includes all metrics, edits, and metadata
   - Optionally strips parameter diffs in non-verbose mode

2. **`format_output_summary()`**: Produces human-readable CLI report
   - Uses visual separators and emoji indicators
   - Shows graph statistics and top edits
   - Includes pass/fail verdict (70% threshold)

=== Usage ===

These functions are called by the CLI tool's main() function based on the `--output` argument.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/compare_workflows.py
* '''Lines:''' L94-255

=== Signatures ===
<syntaxhighlight lang="python">
def format_output_json(
    result: Dict[str, Any],
    metadata: Dict[str, Any],
    verbose: bool = False
) -> str:
    """
    Format result as JSON.

    Args:
        result: Comparison result with similarity_score, edit_cost, top_edits
        metadata: Graph statistics and config info
        verbose: Include parameter diffs in output

    Returns:
        JSON string with indentation
    """

def format_output_summary(
    result: Dict[str, Any],
    metadata: Dict[str, Any],
    verbose: bool = False
) -> str:
    """
    Format result as human-readable summary.

    Args:
        result: Comparison result with similarity_score, edit_cost, top_edits
        metadata: Graph statistics and config info
        verbose: Include detailed parameter diffs

    Returns:
        Multi-line string with formatted report
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.compare_workflows import format_output_json, format_output_summary
</syntaxhighlight>

== I/O Contract ==

=== Inputs (both functions) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| result || Dict[str, Any] || Yes || Comparison result containing `similarity_score`, `edit_cost`, `max_possible_cost`, `top_edits`
|-
| metadata || Dict[str, Any] || Yes || Metadata containing `config_name`, `generated_nodes`, `ground_truth_nodes`, etc.
|-
| verbose || bool || No || If True, include detailed parameter diffs (default: False)
|}

=== Outputs ===
{| class="wikitable"
|-
! Function !! Return Type !! Description
|-
| format_output_json || str || JSON string with 2-space indentation
|-
| format_output_summary || str || Multi-line human-readable report string
|}

== Usage Examples ==

=== JSON Format Implementation ===
<syntaxhighlight lang="python">
def format_output_json(
    result: Dict[str, Any], metadata: Dict[str, Any], verbose: bool = False
) -> str:
    """Format result as JSON"""
    output: Dict[str, Any] = {
        "similarity_score": result["similarity_score"],
        "similarity_percentage": f"{result['similarity_score'] * 100:.1f}%",
        "edit_cost": result["edit_cost"],
        "max_possible_cost": result["max_possible_cost"],
        "top_edits": result["top_edits"],
        "metadata": metadata,
    }

    if verbose:
        metadata_dict = output["metadata"]
        assert isinstance(metadata_dict, dict)
        metadata_dict["verbose"] = True
    else:
        for edit in output["top_edits"]:
            if "parameter_diff" in edit:
                del edit["parameter_diff"]

    return json.dumps(output, indent=2)
</syntaxhighlight>

=== Summary Format Implementation ===
<syntaxhighlight lang="python">
def format_output_summary(
    result: Dict[str, Any], metadata: Dict[str, Any], verbose: bool = False
) -> str:
    """Format result as human-readable summary"""
    lines = []

    # Header
    lines.append("=" * 60)
    lines.append("WORKFLOW COMPARISON SUMMARY")
    lines.append("=" * 60)
    lines.append("")

    # Similarity score
    similarity_pct = result["similarity_score"] * 100
    lines.append(f"Overall Similarity: {similarity_pct:.1f}%")
    lines.append(
        f"Edit Cost:          {result['edit_cost']:.1f} / {result['max_possible_cost']:.1f}"
    )
    lines.append("")

    # Configuration info
    lines.append(f"Configuration: {metadata['config_name']}")
    if metadata.get("config_description"):
        lines.append(f"  {metadata['config_description']}")
    lines.append("")

    # Graph statistics
    lines.append("Graph Statistics:")
    lines.append(
        f"  Generated workflow:    {metadata['generated_nodes']} nodes "
        f"({metadata.get('generated_nodes_after_filter', metadata['generated_nodes'])} after filtering)"
    )
    lines.append(
        f"  Ground truth workflow: {metadata['ground_truth_nodes']} nodes "
        f"({metadata.get('ground_truth_nodes_after_filter', metadata['ground_truth_nodes'])} after filtering)"
    )
    lines.append("")

    # Top edits with priority indicators
    if result["top_edits"]:
        lines.append(f"Top {len(result['top_edits'])} Required Edits:")
        lines.append("-" * 60)

        for i, edit in enumerate(result["top_edits"], 1):
            priority = edit["priority"].upper()
            cost = edit["cost"]
            desc = edit["description"]

            # Priority indicator
            if priority == "CRITICAL":
                indicator = "🔴"
            elif priority == "MAJOR":
                indicator = "🟠"
            else:
                indicator = "🟡"

            lines.append(f"{i}. {indicator} [{priority}] Cost: {cost:.1f}")
            lines.append(f"   {desc}")

            # Add parameter diff if verbose
            if verbose and "parameter_diff" in edit:
                lines.append("")
                lines.extend(
                    _format_parameter_diff(edit["parameter_diff"], indent="   ")
                )

            lines.append("")
    else:
        lines.append("No edits required - workflows are identical!")
        lines.append("")

    # Pass/Fail indicator (70% threshold)
    lines.append("=" * 60)
    if similarity_pct >= 70:
        lines.append("✅ PASS - Workflows are sufficiently similar")
    else:
        lines.append("❌ FAIL - Workflows differ significantly")
    lines.append("=" * 60)

    return "\n".join(lines)
</syntaxhighlight>

=== Example Output: JSON ===
<syntaxhighlight lang="json">
{
  "similarity_score": 0.85,
  "similarity_percentage": "85.0%",
  "edit_cost": 15.0,
  "max_possible_cost": 100.0,
  "top_edits": [
    {
      "type": "node_substitute",
      "description": "Substitute 'HTTP Request' (different parameters)",
      "cost": 5.0,
      "priority": "minor",
      "node_name": "HTTP Request"
    }
  ],
  "metadata": {
    "config_name": "default",
    "generated_nodes": 5,
    "ground_truth_nodes": 5
  }
}
</syntaxhighlight>

=== Example Output: Summary ===
<syntaxhighlight lang="text">
============================================================
WORKFLOW COMPARISON SUMMARY
============================================================

Overall Similarity: 85.0%
Edit Cost:          15.0 / 100.0

Configuration: default

Graph Statistics:
  Generated workflow:    5 nodes (5 after filtering)
  Ground truth workflow: 5 nodes (5 after filtering)

Top 1 Required Edits:
------------------------------------------------------------
1. 🟡 [MINOR] Cost: 5.0
   Substitute 'HTTP Request' (different parameters)

============================================================
✅ PASS - Workflows are sufficiently similar
============================================================
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Output_Formatting]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Workflow_Comparison_Env]]
