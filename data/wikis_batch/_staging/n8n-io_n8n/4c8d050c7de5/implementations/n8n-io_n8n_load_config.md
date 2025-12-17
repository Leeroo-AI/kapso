{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Configuration_Management]], [[domain::Workflow_Analysis]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Concrete tool for loading workflow comparison configuration from multiple sources, provided by the n8n workflow comparison system.

=== Description ===

The `load_config` function provides a flexible configuration loading mechanism that supports:
* Default configuration (when no source is provided)
* Built-in presets (e.g., "preset:strict", "preset:lenient")
* Custom YAML configuration files
* Custom JSON configuration files

This allows users to easily switch between different comparison strategies without modifying code.

=== Usage ===

Use this implementation when you need to:
* Load comparison parameters from different sources
* Apply predefined comparison strategies via presets
* Customize workflow comparison behavior
* Configure node ignoring, similarity weights, and cost functions

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/config_loader.py
* '''Lines:''' L359-389

=== Signature ===
<syntaxhighlight lang="python">
def load_config(config_source: Optional[str] = None) -> WorkflowComparisonConfig:
    """Load configuration from various sources.

    Args:
        config_source: Can be:
            - None: use default config
            - "preset:name": load built-in preset (e.g., "preset:strict")
            - "/path/to/config.yaml": load custom YAML file

    Returns:
        WorkflowComparisonConfig object with loaded settings
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.config_loader import load_config
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| config_source || Optional[str] || No || Configuration source identifier. Can be None (default), "preset:name" for built-in presets, or file path for custom YAML/JSON
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| config || WorkflowComparisonConfig || Configuration object containing all comparison parameters, weights, and cost functions
|}

== Usage Examples ==

=== Using Default Configuration ===
<syntaxhighlight lang="python">
from src.config_loader import load_config

# Load default configuration
config = load_config()

# Use the config for comparison
print(f"Max cost: {config.max_cost}")
print(f"Node substitute cost: {config.node_substitute_cost}")
</syntaxhighlight>

=== Using Built-in Presets ===
<syntaxhighlight lang="python">
# Load strict preset - requires exact matches
strict_config = load_config("preset:strict")

# Load lenient preset - more forgiving comparisons
lenient_config = load_config("preset:lenient")

# Load balanced preset - middle ground
balanced_config = load_config("preset:balanced")

print(f"Strict max cost: {strict_config.max_cost}")
print(f"Lenient max cost: {lenient_config.max_cost}")
</syntaxhighlight>

=== Loading Custom YAML Configuration ===
<syntaxhighlight lang="python">
# Load custom configuration from YAML file
custom_config = load_config("config/my_comparison_rules.yaml")

# Access custom settings
print(f"Ignore nodes: {custom_config.ignore_node_types}")
print(f"Ignore edges: {custom_config.ignore_edge_types}")
</syntaxhighlight>

=== Loading Custom JSON Configuration ===
<syntaxhighlight lang="python">
# Load configuration from JSON file
json_config = load_config("config/comparison_config.json")

# Use for workflow comparison
from src.similarity import calculate_graph_edit_distance

result = calculate_graph_edit_distance(graph1, graph2, json_config)
</syntaxhighlight>

=== Configuration-Driven Comparison ===
<syntaxhighlight lang="python">
def compare_workflows_with_config(workflow1_path, workflow2_path, config_source):
    """Compare workflows using specified configuration."""
    from src.compare_workflows import load_workflow
    from src.graph_builder import build_workflow_graph
    from src.similarity import calculate_graph_edit_distance

    # Load configuration
    config = load_config(config_source)

    # Load workflows
    wf1 = load_workflow(workflow1_path)
    wf2 = load_workflow(workflow2_path)

    # Build graphs with config
    g1 = build_workflow_graph(wf1, config)
    g2 = build_workflow_graph(wf2, config)

    # Calculate similarity with config
    result = calculate_graph_edit_distance(g1, g2, config)

    return result

# Use with different configurations
strict_result = compare_workflows_with_config(
    "wf1.json",
    "wf2.json",
    "preset:strict"
)

lenient_result = compare_workflows_with_config(
    "wf1.json",
    "wf2.json",
    "preset:lenient"
)

print(f"Strict similarity: {strict_result['similarity_score']:.2f}")
print(f"Lenient similarity: {lenient_result['similarity_score']:.2f}")
</syntaxhighlight>

== Configuration Options ==

=== Preset Types ===
{| class="wikitable"
|-
! Preset !! Description !! Use Case
|-
| strict || High precision, low tolerance for differences || Exact workflow validation, regression testing
|-
| lenient || High tolerance for differences || Fuzzy matching, workflow similarity detection
|-
| balanced || Moderate tolerance || General purpose comparison
|}

=== Custom Configuration Format ===
Example YAML configuration structure:
<syntaxhighlight lang="yaml">
max_cost: 100
node_insert_cost: 10
node_delete_cost: 10
node_substitute_cost: 5
edge_insert_cost: 5
edge_delete_cost: 5

ignore_node_types:
  - "n8n-nodes-base.stickyNote"

ignore_edge_types:
  - "annotation"
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Configuration_Loading]]

=== Used By ===
* [[used_by::Implementation:n8n-io_n8n_build_workflow_graph]]
* [[used_by::Implementation:n8n-io_n8n_calculate_graph_edit_distance]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Workflow_Comparison]]

[[Category:Implementation]]
[[Category:n8n]]
[[Category:Configuration]]
