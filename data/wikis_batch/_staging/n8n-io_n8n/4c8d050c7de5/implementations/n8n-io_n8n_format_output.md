{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Output_Formatting]], [[domain::CLI_Tools]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Concrete tools for formatting workflow comparison results in multiple output formats, provided by the n8n workflow comparison system.

=== Description ===

The output formatting functions transform comparison results into user-friendly formats:
* '''JSON format''' (`format_output_json`): Structured data for programmatic consumption, integration with other tools, and automated testing
* '''Summary format''' (`format_output_summary`): Human-readable text with color-coded PASS/FAIL verdicts, ideal for command-line usage and reporting

Both formatters present the same underlying data but optimize for different use cases and audiences.

=== Usage ===

Use these implementations when you need to:
* Display comparison results to users in readable formats
* Export results for integration with CI/CD pipelines
* Generate reports for workflow validation
* Provide color-coded visual feedback in terminals

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
    """Format result as JSON.

    Args:
        result: Comparison result from calculate_graph_edit_distance
        metadata: Additional metadata (workflow names, paths, etc.)
        verbose: Include detailed operations in output

    Returns:
        JSON-formatted string
    """

def format_output_summary(
    result: Dict[str, Any],
    metadata: Dict[str, Any],
    verbose: bool = False
) -> str:
    """Format result as human-readable summary.

    Args:
        result: Comparison result from calculate_graph_edit_distance
        metadata: Additional metadata (workflow names, paths, etc.)
        verbose: Include detailed operations in output

    Returns:
        Formatted summary string with color codes
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.compare_workflows import format_output_json, format_output_summary
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| result || Dict[str, Any] || Yes || Comparison result containing similarity_score, edit_cost, operations, etc.
|-
| metadata || Dict[str, Any] || Yes || Workflow metadata (names, paths, node counts)
|-
| verbose || bool || No || Include detailed operation list (default: False)
|}

=== Outputs ===
{| class="wikitable"
|-
! Function !! Return Type !! Description
|-
| format_output_json || str || JSON-formatted comparison result
|-
| format_output_summary || str || Human-readable text with ANSI color codes
|}

=== Result Structure (Input) ===
Expected result dictionary structure:
<syntaxhighlight lang="python">
{
    "similarity_score": 0.85,
    "edit_cost": 15.0,
    "max_cost": 100.0,
    "operations": [
        {"type": "node_insert", "description": "...", "cost": 10.0, "details": {...}},
        # ... more operations
    ],
    "node_counts": {"g1": 5, "g2": 6},
    "edge_counts": {"g1": 4, "g2": 5}
}
</syntaxhighlight>

=== Metadata Structure (Input) ===
Expected metadata dictionary structure:
<syntaxhighlight lang="python">
{
    "workflow1_name": "Reference Workflow",
    "workflow2_name": "Test Workflow",
    "workflow1_path": "/path/to/workflow1.json",
    "workflow2_path": "/path/to/workflow2.json"
}
</syntaxhighlight>

== Usage Examples ==

=== JSON Output Format ===
<syntaxhighlight lang="python">
from src.similarity import calculate_graph_edit_distance
from src.compare_workflows import format_output_json

# Calculate comparison
result = calculate_graph_edit_distance(graph1, graph2, config)

# Prepare metadata
metadata = {
    "workflow1_name": "Production Workflow",
    "workflow2_name": "Test Workflow",
    "workflow1_path": "workflows/prod.json",
    "workflow2_path": "workflows/test.json"
}

# Format as JSON (non-verbose)
json_output = format_output_json(result, metadata, verbose=False)
print(json_output)

# Output:
# {
#   "similarity_score": 0.85,
#   "similarity_percentage": "85.0%",
#   "edit_cost": 15.0,
#   "max_cost": 100.0,
#   "verdict": "PASS",
#   "workflow1": {
#     "name": "Production Workflow",
#     "path": "workflows/prod.json",
#     "nodes": 5,
#     "edges": 4
#   },
#   "workflow2": {
#     "name": "Test Workflow",
#     "path": "workflows/test.json",
#     "nodes": 6,
#     "edges": 5
#   }
# }
</syntaxhighlight>

=== JSON Output with Verbose Details ===
<syntaxhighlight lang="python">
# Format as JSON with operations included
json_output_verbose = format_output_json(result, metadata, verbose=True)
print(json_output_verbose)

# Output includes additional "operations" array:
# {
#   "similarity_score": 0.85,
#   "similarity_percentage": "85.0%",
#   "edit_cost": 15.0,
#   "max_cost": 100.0,
#   "verdict": "PASS",
#   "operations": [
#     {
#       "type": "node_insert",
#       "description": "Add missing node 'HTTP Request'",
#       "cost": 10.0,
#       "details": {...}
#     },
#     {
#       "type": "edge_insert",
#       "description": "Add missing connection: Webhook -> HTTP Request",
#       "cost": 5.0,
#       "details": {...}
#     }
#   ],
#   "workflow1": {...},
#   "workflow2": {...}
# }
</syntaxhighlight>

=== Summary Output Format ===
<syntaxhighlight lang="python">
from src.compare_workflows import format_output_summary

# Format as human-readable summary
summary_output = format_output_summary(result, metadata, verbose=False)
print(summary_output)

# Output (with color codes in terminal):
# ================================================================================
# Workflow Comparison Result: PASS
# ================================================================================
#
# Similarity Score: 85.0%
# Edit Cost: 15.0 / 100.0
#
# Workflow 1: Production Workflow
#   Path: workflows/prod.json
#   Nodes: 5, Edges: 4
#
# Workflow 2: Test Workflow
#   Path: workflows/test.json
#   Nodes: 6, Edges: 5
#
# ================================================================================
</syntaxhighlight>

=== Summary Output with Verbose Details ===
<syntaxhighlight lang="python">
# Format as summary with operations
summary_verbose = format_output_summary(result, metadata, verbose=True)
print(summary_verbose)

# Output includes operations section:
# ... (header same as above) ...
#
# Edit Operations (2):
# ----------------------------------------
#   [node_insert] Add missing node 'HTTP Request' (type: n8n-nodes-base.httpRequest)
#     Cost: 10.0
#
#   [edge_insert] Add missing connection: Webhook -> HTTP Request
#     Cost: 5.0
#
# ================================================================================
</syntaxhighlight>

=== Selecting Format Based on Output Destination ===
<syntaxhighlight lang="python">
import sys
import json

def output_comparison_result(result, metadata, output_format="auto", verbose=False):
    """
    Output comparison result in specified format.

    Args:
        result: Comparison result
        metadata: Workflow metadata
        output_format: "json", "summary", or "auto" (auto-detect based on stdout)
        verbose: Include detailed operations
    """
    # Auto-detect format
    if output_format == "auto":
        # Use JSON if stdout is redirected (piped or file)
        if not sys.stdout.isatty():
            output_format = "json"
        else:
            output_format = "summary"

    # Format and output
    if output_format == "json":
        output = format_output_json(result, metadata, verbose)
    else:
        output = format_output_summary(result, metadata, verbose)

    print(output)

# Usage
output_comparison_result(result, metadata, output_format="auto", verbose=True)
</syntaxhighlight>

=== Saving JSON Output to File ===
<syntaxhighlight lang="python">
import json

# Format as JSON
json_output = format_output_json(result, metadata, verbose=True)

# Parse and save
output_data = json.loads(json_output)

with open("comparison_result.json", "w") as f:
    json.dump(output_data, f, indent=2)

print("Comparison result saved to comparison_result.json")

# Later: Load and analyze
with open("comparison_result.json", "r") as f:
    loaded_result = json.load(f)

print(f"Loaded similarity: {loaded_result['similarity_score']:.2%}")
print(f"Verdict: {loaded_result['verdict']}")
</syntaxhighlight>

=== Colored Terminal Output ===
<syntaxhighlight lang="python">
from src.compare_workflows import format_output_summary

# Format summary (includes ANSI color codes)
summary = format_output_summary(result, metadata, verbose=False)

# Display in terminal with colors
print(summary)

# The summary includes color codes:
# - Green for PASS verdict
# - Red for FAIL verdict
# - Color-coded similarity percentage

# To strip colors for logging:
import re
ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
plain_summary = ansi_escape.sub('', summary)

with open("comparison.log", "w") as f:
    f.write(plain_summary)
</syntaxhighlight>

=== Integration with CI/CD Pipeline ===
<syntaxhighlight lang="python">
#!/usr/bin/env python3
"""CI/CD workflow comparison script."""

import sys
import json
from src.compare_workflows import load_workflow, format_output_json
from src.graph_builder import build_workflow_graph
from src.similarity import calculate_graph_edit_distance
from src.config_loader import load_config

def main():
    # Load workflows
    reference = load_workflow("workflows/reference.json")
    candidate = load_workflow("workflows/candidate.json")

    # Build graphs
    config = load_config("preset:strict")
    g1 = build_workflow_graph(reference, config)
    g2 = build_workflow_graph(candidate, config)

    # Compare
    result = calculate_graph_edit_distance(g1, g2, config)

    # Prepare metadata
    metadata = {
        "workflow1_name": reference.get("name", "Reference"),
        "workflow2_name": candidate.get("name", "Candidate"),
        "workflow1_path": "workflows/reference.json",
        "workflow2_path": "workflows/candidate.json"
    }

    # Output as JSON for CI/CD
    json_output = format_output_json(result, metadata, verbose=True)
    print(json_output)

    # Exit with appropriate code
    similarity = result['similarity_score']
    threshold = 0.8

    if similarity >= threshold:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure

if __name__ == "__main__":
    main()
</syntaxhighlight>

=== Batch Comparison Reporting ===
<syntaxhighlight lang="python">
import json
from pathlib import Path

def generate_batch_report(comparison_results, output_file="batch_report.json"):
    """
    Generate comprehensive report for multiple comparisons.

    Args:
        comparison_results: List of (result, metadata) tuples
        output_file: Output file path
    """
    batch_report = {
        "total_comparisons": len(comparison_results),
        "passed": 0,
        "failed": 0,
        "comparisons": []
    }

    for result, metadata in comparison_results:
        # Format each comparison as JSON
        json_str = format_output_json(result, metadata, verbose=False)
        comparison_data = json.loads(json_str)

        batch_report["comparisons"].append(comparison_data)

        if comparison_data["verdict"] == "PASS":
            batch_report["passed"] += 1
        else:
            batch_report["failed"] += 1

    # Calculate summary statistics
    similarities = [c["similarity_score"] for c in batch_report["comparisons"]]
    batch_report["statistics"] = {
        "average_similarity": sum(similarities) / len(similarities),
        "min_similarity": min(similarities),
        "max_similarity": max(similarities),
    }

    # Save report
    with open(output_file, "w") as f:
        json.dump(batch_report, f, indent=2)

    print(f"Batch report saved to {output_file}")
    print(f"  Total: {batch_report['total_comparisons']}")
    print(f"  Passed: {batch_report['passed']}")
    print(f"  Failed: {batch_report['failed']}")
    print(f"  Average similarity: {batch_report['statistics']['average_similarity']:.2%}")

# Usage
results = []
for wf_pair in workflow_pairs:
    result = compare_workflows(wf_pair[0], wf_pair[1], config)
    metadata = create_metadata(wf_pair)
    results.append((result, metadata))

generate_batch_report(results, "batch_report.json")
</syntaxhighlight>

== Output Format Specifications ==

=== JSON Output Schema ===
<syntaxhighlight lang="json">
{
  "similarity_score": <float>,
  "similarity_percentage": <string>,
  "edit_cost": <float>,
  "max_cost": <float>,
  "verdict": "PASS" | "FAIL",
  "workflow1": {
    "name": <string>,
    "path": <string>,
    "nodes": <int>,
    "edges": <int>
  },
  "workflow2": {
    "name": <string>,
    "path": <string>,
    "nodes": <int>,
    "edges": <int>
  },
  "operations": [  // Only if verbose=True
    {
      "type": <string>,
      "description": <string>,
      "cost": <float>,
      "details": <object>
    }
  ]
}
</syntaxhighlight>

=== Summary Output Format ===
<pre>
================================================================================
Workflow Comparison Result: PASS/FAIL
================================================================================

Similarity Score: XX.X%
Edit Cost: XX.X / XXX.X

Workflow 1: <name>
  Path: <path>
  Nodes: X, Edges: X

Workflow 2: <name>
  Path: <path>
  Nodes: X, Edges: X

[If verbose:]
Edit Operations (X):
----------------------------------------
  [<type>] <description>
    Cost: X.X

================================================================================
</pre>

=== Color Codes (Summary Format) ===
{| class="wikitable"
|-
! Element !! Color !! ANSI Code !! Condition
|-
| PASS verdict || Green || \033[92m || similarity >= 0.8
|-
| FAIL verdict || Red || \033[91m || similarity < 0.8
|-
| Similarity percentage || Green/Red || \033[92m/\033[91m || Based on verdict
|-
| Headers || Bold || \033[1m || Always
|}

== Design Considerations ==

=== Why Two Formats? ===
* '''JSON:''' Machine-readable, parseable, ideal for automation
* '''Summary:''' Human-readable, visual, ideal for interactive use

=== Verbose Mode ===
* '''Non-verbose:''' Quick overview, suitable for dashboards
* '''Verbose:''' Detailed analysis, suitable for debugging and troubleshooting

=== Pass/Fail Threshold ===
The default threshold for PASS/FAIL verdict is 0.8 (80% similarity), but this can be adjusted based on use case requirements.

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Result_Formatting]]

=== Uses ===
* [[uses::Implementation:n8n-io_n8n_calculate_graph_edit_distance]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Workflow_Comparison]]

[[Category:Implementation]]
[[Category:n8n]]
[[Category:Output_Formatting]]
