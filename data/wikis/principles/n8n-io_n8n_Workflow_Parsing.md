# Principle: Workflow Parsing

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
* [[source::Doc|n8n Workflow JSON|https://docs.n8n.io]]
|-
! Domains
| [[domain::Workflow_Evaluation]], [[domain::JSON_Parsing]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Principle for loading and validating n8n workflow JSON files for comparison, with proper error handling for common issues.

=== Description ===

Workflow Parsing handles the input phase of comparison:

1. **File Loading**: Reads JSON file from filesystem
2. **JSON Parsing**: Deserializes to Python dictionary
3. **Error Handling**: Provides clear errors for missing files or invalid JSON
4. **Structure Validation**: Expects `nodes` and `connections` keys

The parsed workflow contains:
- **nodes**: Array of node objects with name, type, parameters
- **connections**: Nested structure mapping source nodes to targets

=== Usage ===

Apply this principle when:
- Loading workflow definitions for processing
- Building workflow import/export tools
- Implementing workflow validation utilities
- Creating comparison or analysis tools

== Theoretical Basis ==

Workflow parsing follows a **Load-or-Fail** pattern:

<syntaxhighlight lang="python">
# Pseudo-code for workflow parsing

def load_workflow(path: str) -> Dict[str, Any]:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Workflow file not found: {path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {path}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        sys.exit(1)
</syntaxhighlight>

n8n workflow JSON structure:
```json
{
  "name": "My Workflow",
  "nodes": [
    {"name": "Start", "type": "n8n-nodes-base.manualTrigger", ...},
    {"name": "HTTP", "type": "n8n-nodes-base.httpRequest", ...}
  ],
  "connections": {
    "Start": {
      "main": [[{"node": "HTTP", "type": "main", "index": 0}]]
    }
  }
}
```

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_load_workflow]]
