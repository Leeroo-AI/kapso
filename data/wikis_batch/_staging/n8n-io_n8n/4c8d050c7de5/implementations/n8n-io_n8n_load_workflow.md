{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Workflow_Analysis]], [[domain::File_IO]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Concrete tool for loading n8n workflow JSON files from the filesystem, provided by the n8n workflow comparison system.

=== Description ===

The `load_workflow` function reads and parses workflow JSON files from disk, handling common error cases like missing files and malformed JSON. It provides clear error messages and exits gracefully on failures, making it suitable for command-line tools.

=== Usage ===

Use this implementation when you need to:
* Load n8n workflow definitions from JSON files
* Validate that workflow files exist and contain valid JSON
* Provide user-friendly error messages for file loading issues
* Prepare workflow data for comparison or analysis

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/compare_workflows.py
* '''Lines:''' L67-91

=== Signature ===
<syntaxhighlight lang="python">
def load_workflow(path: str) -> Dict[str, Any]:
    """Load workflow JSON from file."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.compare_workflows import load_workflow
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| path || str || Yes || Absolute or relative path to the workflow JSON file
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| workflow || Dict[str, Any] || Parsed workflow data structure containing nodes, connections, and metadata
|}

=== Error Handling ===
{| class="wikitable"
|-
! Error Type !! Behavior
|-
| FileNotFoundError || Prints error message to stderr and exits with code 1
|-
| json.JSONDecodeError || Prints JSON parsing error details to stderr and exits with code 1
|}

== Usage Examples ==

=== Basic Workflow Loading ===
<syntaxhighlight lang="python">
import json
from pathlib import Path

# Load a workflow from a file
workflow = load_workflow("workflows/my_workflow.json")

# Access workflow properties
print(f"Workflow name: {workflow.get('name', 'Unnamed')}")
print(f"Number of nodes: {len(workflow.get('nodes', []))}")
print(f"Number of connections: {len(workflow.get('connections', {}))}")
</syntaxhighlight>

=== Loading Multiple Workflows for Comparison ===
<syntaxhighlight lang="python">
# Load two workflows for comparison
workflow1 = load_workflow("workflows/version1.json")
workflow2 = load_workflow("workflows/version2.json")

# The workflows are now ready for graph construction and comparison
print(f"Comparing: {workflow1.get('name')} vs {workflow2.get('name')}")
</syntaxhighlight>

=== Error Handling Example ===
<syntaxhighlight lang="python">
# This function handles errors internally and exits
# If the file doesn't exist, it will print an error and exit
try:
    workflow = load_workflow("nonexistent.json")
except SystemExit:
    # The function already printed the error message
    print("Workflow loading failed, cleanup code here")
</syntaxhighlight>

== Implementation Notes ==

=== File Format ===
The function expects n8n workflow JSON files with the following structure:
<syntaxhighlight lang="json">
{
  "name": "Workflow Name",
  "nodes": [...],
  "connections": {...},
  "settings": {...}
}
</syntaxhighlight>

=== Error Messages ===
Error messages are written to stderr to allow proper separation of output and errors in command-line pipelines.

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Workflow_Loading]]

=== Used By ===
* [[used_by::Implementation:n8n-io_n8n_build_workflow_graph]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Workflow_Comparison]]

[[Category:Implementation]]
[[Category:n8n]]
[[Category:Workflow_Analysis]]
