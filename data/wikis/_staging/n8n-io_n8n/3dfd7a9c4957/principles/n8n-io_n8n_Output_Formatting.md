# Principle: Output Formatting

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

Principle for transforming comparison results into consumable formats (JSON for automation, human-readable summary for CLI).

=== Description ===

Output Formatting provides multiple representations of comparison results:

1. **JSON Format**: Machine-readable output with full structure
2. **Summary Format**: Human-readable report with visual indicators
3. **Verbose Mode**: Optional detailed parameter diffs
4. **Pass/Fail Indicator**: Clear threshold-based verdict (70% similarity)

Key design decisions:
- JSON preserves all data for programmatic consumption
- Summary uses emoji indicators for quick visual scanning
- Parameter diffs are truncated for readability
- Consistent structure across formats enables tooling

=== Usage ===

Apply this principle when:
- Building CLI tools with multiple output format requirements
- Creating reports for both human and machine consumption
- Implementing evaluation pipelines with threshold checks
- Designing user-facing comparison tools

== Theoretical Basis ==

The dual-format approach follows the **Reporter Pattern**:

<syntaxhighlight lang="python">
# JSON format - full data, machine-readable
def format_output_json(result, metadata, verbose=False):
    output = {
        "similarity_score": result["similarity_score"],
        "similarity_percentage": f"{result['similarity_score'] * 100:.1f}%",
        "edit_cost": result["edit_cost"],
        "max_possible_cost": result["max_possible_cost"],
        "top_edits": result["top_edits"],
        "metadata": metadata,
    }
    # Optionally strip parameter diffs for concise output
    if not verbose:
        for edit in output["top_edits"]:
            if "parameter_diff" in edit:
                del edit["parameter_diff"]
    return json.dumps(output, indent=2)

# Summary format - human-readable with visual indicators
def format_output_summary(result, metadata, verbose=False):
    lines = []
    # Header with separators
    lines.append("=" * 60)
    lines.append("WORKFLOW COMPARISON SUMMARY")

    # Priority indicators using emoji
    for edit in result["top_edits"]:
        if edit["priority"] == "critical":
            indicator = "🔴"
        elif edit["priority"] == "major":
            indicator = "🟠"
        else:
            indicator = "🟡"

    # Pass/Fail verdict (70% threshold)
    if result["similarity_score"] * 100 >= 70:
        lines.append("✅ PASS")
    else:
        lines.append("❌ FAIL")

    return "\n".join(lines)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_format_output]]
