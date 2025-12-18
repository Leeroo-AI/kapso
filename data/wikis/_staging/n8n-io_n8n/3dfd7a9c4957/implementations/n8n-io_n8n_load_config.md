# Implementation: load_config

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Configuration]], [[domain::Workflow_Evaluation]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Concrete function for loading workflow comparison configuration from multiple sources: defaults, presets, or custom files.

=== Description ===

`load_config()` is the main entry point for obtaining comparison configuration:

1. **None**: Returns default `WorkflowComparisonConfig()` with sensible defaults
2. **"preset:name"**: Loads named preset from built-in YAML files
3. **File path**: Loads configuration from YAML (.yaml/.yml) or JSON (.json) file

The function delegates to `WorkflowComparisonConfig` class methods for actual parsing.

=== Usage ===

Call this function at the start of workflow comparison to obtain configuration. CLI tools can accept `--preset` or `--config` arguments to specify source.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/config_loader.py
* '''Lines:''' L359-389

=== Signature ===
<syntaxhighlight lang="python">
def load_config(config_source: Optional[str] = None) -> WorkflowComparisonConfig:
    """
    Load configuration from various sources.

    Args:
        config_source: Can be:
            - None: use default config
            - "preset:name": load built-in preset (e.g., "preset:strict")
            - "/path/to/config.yaml": load custom YAML file
            - "/path/to/config.json": load custom JSON file

    Returns:
        WorkflowComparisonConfig instance

    Raises:
        ValueError: If preset not found or file format unsupported.
        FileNotFoundError: If config file doesn't exist.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.config_loader import load_config, WorkflowComparisonConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| config_source || Optional[str] || No || Configuration source specification
|}

=== Outputs ===
{| class="wikitable"
|-
! Return Type !! Description
|-
| WorkflowComparisonConfig || Complete configuration dataclass instance
|}

=== WorkflowComparisonConfig Key Fields ===
{| class="wikitable"
|-
! Field !! Type !! Default !! Description
|-
| node_insertion_cost || float || 10.0 || Cost for adding a node
|-
| node_deletion_cost || float || 10.0 || Cost for removing a node
|-
| node_substitution_same_type || float || 1.0 || Cost for parameter-only changes
|-
| node_substitution_different_type || float || 15.0 || Cost for type change
|-
| edge_insertion_cost || float || 5.0 || Cost for adding a connection
|-
| edge_deletion_cost || float || 5.0 || Cost for removing a connection
|-
| similarity_groups || Dict[str, List[str]] || {} || Node types considered equivalent
|-
| ignored_node_types || Set[str] || set() || Node types to skip
|-
| max_edits || int || 15 || Maximum edits to report
|}

== Usage Examples ==

=== Default Configuration ===
<syntaxhighlight lang="python">
from src.config_loader import load_config

# Use default settings
config = load_config()
# Returns WorkflowComparisonConfig with default costs
</syntaxhighlight>

=== Preset Configuration ===
<syntaxhighlight lang="python">
# Load strict preset (higher penalties)
config = load_config("preset:strict")

# Load lenient preset (lower penalties)
config = load_config("preset:lenient")

# Load standard preset
config = load_config("preset:standard")
</syntaxhighlight>

=== Custom File Configuration ===
<syntaxhighlight lang="python">
# Load from YAML
config = load_config("/path/to/my-config.yaml")

# Load from JSON
config = load_config("/path/to/my-config.json")
</syntaxhighlight>

=== CLI Usage ===
<syntaxhighlight lang="python">
# In compare_workflows.py
args = parse_args()

if args.config:
    config = load_config(args.config)
elif args.preset:
    config = load_config(f"preset:{args.preset}")
else:
    config = load_config()  # Default
</syntaxhighlight>

=== Custom YAML Example ===
<syntaxhighlight lang="yaml">
# my-config.yaml
version: "1.0"
name: "custom"
description: "Custom comparison rules"

costs:
  nodes:
    insertion: 15.0
    deletion: 15.0
    substitution:
      same_type: 2.0
      similar_type: 8.0
      different_type: 20.0
      trigger_mismatch: 100.0

similarity_groups:
  http_nodes:
    - "n8n-nodes-base.httpRequest"
    - "n8n-nodes-base.http"

ignore:
  node_types:
    - "n8n-nodes-base.stickyNote"
  global_parameters:
    - "position"
    - "id"
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Comparison_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Workflow_Comparison_Env]]
