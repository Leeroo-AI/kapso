# Implementation: load_workflow

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Workflow_Evaluation]], [[domain::JSON_Parsing]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Concrete function for loading n8n workflow JSON files with comprehensive error handling.

=== Description ===

`load_workflow()` loads a workflow JSON file and returns it as a dictionary:

1. Opens the file at the given path
2. Parses JSON content using `json.load()`
3. Returns the parsed dictionary
4. Exits with clear error message on any failure

The function is designed for CLI usage, calling `sys.exit(1)` on errors rather than raising exceptions.

=== Usage ===

Call this function with a path to a workflow JSON file. It's used to load both generated and ground truth workflows for comparison.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/compare_workflows.py
* '''Lines:''' L67-91

=== Signature ===
<syntaxhighlight lang="python">
def load_workflow(path: str) -> Dict[str, Any]:
    """
    Load workflow JSON from file.

    Args:
        path: Path to workflow JSON file

    Returns:
        Workflow dictionary

    Raises:
        SystemExit: If file cannot be loaded
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Usually not imported - internal to compare_workflows.py
# For library usage, access via the module
from src.compare_workflows import load_workflow
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| path || str || Yes || File path to workflow JSON
|}

=== Outputs ===
{| class="wikitable"
|-
! Return Type !! Description
|-
| Dict[str, Any] || Parsed workflow dictionary with "nodes" and "connections" keys
|}

=== Error Cases ===
{| class="wikitable"
|-
! Error !! Output !! Exit Code
|-
| FileNotFoundError || "Error: Workflow file not found: {path}" || 1
|-
| json.JSONDecodeError || "Error: Invalid JSON in {path}: {details}" || 1
|-
| Other Exception || "Error loading {path}: {details}" || 1
|}

== Usage Examples ==

=== Basic Loading ===
<syntaxhighlight lang="python">
from src.compare_workflows import load_workflow

# Load generated workflow
generated = load_workflow("./generated.json")

# Load ground truth workflow
ground_truth = load_workflow("./expected.json")

# Access workflow components
nodes = generated.get("nodes", [])
connections = generated.get("connections", {})

print(f"Workflow has {len(nodes)} nodes")
</syntaxhighlight>

=== Workflow Structure ===
<syntaxhighlight lang="python">
workflow = load_workflow("example.json")

# Example structure:
# {
#   "name": "Email Processor",
#   "nodes": [
#     {
#       "name": "Gmail Trigger",
#       "type": "n8n-nodes-base.gmailTrigger",
#       "typeVersion": 1,
#       "position": [250, 300],
#       "parameters": {
#         "pollInterval": 5,
#         "filters": {"includeSpam": false}
#       }
#     },
#     {
#       "name": "Send Slack",
#       "type": "n8n-nodes-base.slack",
#       "typeVersion": 2,
#       "position": [450, 300],
#       "parameters": {
#         "channel": "#alerts",
#         "text": "New email received"
#       }
#     }
#   ],
#   "connections": {
#     "Gmail Trigger": {
#       "main": [
#         [{"node": "Send Slack", "type": "main", "index": 0}]
#       ]
#     }
#   }
# }
</syntaxhighlight>

=== CLI Usage ===
<syntaxhighlight lang="python">
# In main() of compare_workflows.py
def main():
    args = parse_args()

    # Load both workflows
    generated = load_workflow(args.generated)
    ground_truth = load_workflow(args.ground_truth)

    # Build graphs and compare...
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Workflow_Parsing]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Workflow_Comparison_Env]]
