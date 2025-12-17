{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|JSON Format|https://www.json.org]]
* [[source::Paper|Human-Computer Interaction|https://en.wikipedia.org/wiki/Human%E2%80%93computer_interaction]]
|-
! Domains
| [[domain::Data_Formatting]], [[domain::Workflow_Analysis]], [[domain::User_Interface]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Result Formatting is the principle of presenting workflow comparison results in both machine-readable JSON format and human-readable summary format, each optimized for their respective audiences.

=== Description ===

The result formatting principle recognizes that workflow comparison output serves two distinct audiences with different needs:

1. **Machine Consumers**: Automated systems, APIs, and downstream processing tools that need structured, parseable data with complete information

2. **Human Consumers**: Developers, analysts, and workflow designers who need concise, readable summaries highlighting key differences

Both formats include the same core information but organize and present it differently:

**JSON Format Features**:
* Complete structured data with all fields
* Easily parsed by programming languages
* Suitable for storage in databases
* Enables automated processing pipelines
* Machine-readable operation lists

**Summary Format Features**:
* Concise, natural language descriptions
* Visual indicators (bullet points, indentation)
* Prioritized information (most important first)
* Human-friendly value representations
* Contextual explanations

By providing both formats, the system supports integration into automated workflows while remaining accessible to human users who need to understand and act on the comparison results.

=== Usage ===

Apply this principle when:
* Building APIs that serve both humans and machines
* Creating reports that need to be both actionable and archivable
* Designing systems with multiple output consumers
* Supporting both interactive and batch processing use cases
* Enabling both manual review and automated decision-making

== Theoretical Basis ==

=== Dual Format Pattern ===

```
Input → Processing → {
    JSON Output    (machine-readable)
    Summary Output (human-readable)
}
```

=== JSON Format Structure ===

```json
{
  "comparison_metadata": {
    "workflow1_path": "/path/to/workflow1.json",
    "workflow2_path": "/path/to/workflow2.json",
    "config_preset": "standard",
    "timestamp": "2025-12-17T20:00:00Z"
  },
  "similarity": {
    "score": 0.85,
    "edit_cost": 2.5,
    "max_cost": 16.0
  },
  "graph_info": {
    "workflow1": {
      "nodes": 8,
      "edges": 10,
      "triggers": 1
    },
    "workflow2": {
      "nodes": 8,
      "edges": 10,
      "triggers": 1
    }
  },
  "operations": [
    {
      "type": "substitute_node",
      "source_node": "node_2",
      "target_node": "node_2",
      "changes": [
        {
          "type": "parameter_change",
          "parameter": "url",
          "old_value": "https://api.example.com/v1",
          "new_value": "https://api.example.com/v2"
        }
      ],
      "cost": 0.1
    }
  ]
}
```

=== Summary Format Structure ===

```
Workflow Comparison Summary
===========================

Similarity Score: 0.85 (85% similar)

Workflows:
  - Workflow 1: /path/to/workflow1.json (8 nodes, 10 connections)
  - Workflow 2: /path/to/workflow2.json (8 nodes, 10 connections)

Configuration: standard preset

Differences Found: 3 operations (total cost: 2.5)

1. Node Substitution (node_2)
   - Parameter changed: url
     • Old value: https://api.example.com/v1
     • New value: https://api.example.com/v2
   - Cost: 0.1

2. Node Insertion (node_7)
   - Type: n8n-nodes-base.set
   - Cost: 1.0

3. Edge Deletion (node_4 → node_5)
   - Cost: 0.5
```

=== Format Selection Logic ===

```python
def format_output(results, format_type):
    if format_type == 'json':
        return json.dumps(results, indent=2)
    elif format_type == 'summary':
        return generate_human_summary(results)
    elif format_type == 'both':
        return {
            'json': json.dumps(results, indent=2),
            'summary': generate_human_summary(results)
        }
```

=== Human-Friendly Value Formatting ===

Transform technical values for readability:

```python
def format_value(value):
    # Truncate long strings
    if isinstance(value, str) and len(value) > 50:
        return value[:47] + "..."

    # Pretty-print JSON objects
    if isinstance(value, dict):
        return json.dumps(value, indent=2)

    # Format numbers
    if isinstance(value, float):
        return f"{value:.2f}"

    return str(value)
```

=== Progressive Disclosure ===

Summary format uses progressive disclosure:

1. **Top-level**: Similarity score (most important metric)
2. **Second-level**: Workflow metadata and configuration
3. **Third-level**: Detailed operation list
4. **Fourth-level**: Individual parameter changes

Users can stop reading when they have sufficient information.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_format_output]]

=== Related Principles ===
* [[related::Principle:n8n-io_n8n_Similarity_Calculation]]
* [[related::Principle:n8n-io_n8n_Edit_Extraction]]
