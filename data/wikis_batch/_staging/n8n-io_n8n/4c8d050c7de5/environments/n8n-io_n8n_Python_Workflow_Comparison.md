# Environment: Python Workflow Comparison

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n Workflow Comparison|https://github.com/n8n-io/n8n/tree/master/packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python]]
* [[source::Doc|pyproject.toml|packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/pyproject.toml]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Workflow_Automation]], [[domain::Graph_Algorithms]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

Python 3.11+ environment with NetworkX, NumPy, and SciPy for graph-based workflow similarity comparison using graph edit distance algorithms.

=== Description ===

This environment provides the runtime context for the n8n workflow comparison tool, which evaluates AI-generated n8n workflows against ground truth using graph edit distance (GED) algorithms. It converts n8n workflow JSON files into NetworkX directed graphs and calculates structural similarity scores. The environment requires scientific computing libraries for efficient graph operations.

=== Usage ===

Use this environment for running the **Workflow Comparison** evaluation tool, which compares n8n workflows using configurable cost functions. It is required for all implementations in the Workflow Comparison workflow.

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Any (cross-platform) || No OS-specific requirements
|-
| Python || >= 3.11 || Required for type annotations
|-
| Memory || Depends on workflow size || GED algorithm is computationally intensive for large graphs
|}

== Dependencies ==

=== Python Packages ===

* `networkx` >= 3.2 (Graph data structure and GED algorithms)
* `numpy` >= 2.3.4 (Numerical operations)
* `pyyaml` >= 6.0 (YAML configuration file parsing)
* `scipy` >= 1.16.3 (Scientific computing)

=== Development Dependencies ===

* `pytest` >= 9.0.1
* `pytest-cov` >= 7.0.0
* `ruff` >= 0.14.5
* `hatchling` (build system)

== Credentials ==

No credentials required. This is a CLI tool that processes local workflow JSON files.

== Quick Install ==

<syntaxhighlight lang="bash">
# Requires Python 3.11+
pip install networkx>=3.2 numpy>=2.3.4 pyyaml>=6.0 scipy>=1.16.3

# Or install from package directory
pip install -e packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python
</syntaxhighlight>

== Code Evidence ==

Python version requirement from `pyproject.toml:5`:
<syntaxhighlight lang="toml">
requires-python = ">=3.11"
</syntaxhighlight>

Dependencies from `pyproject.toml:6-11`:
<syntaxhighlight lang="toml">
dependencies = [
    "networkx>=3.2",
    "numpy>=2.3.4",
    "pyyaml>=6.0",
    "scipy>=1.16.3",
]
</syntaxhighlight>

NetworkX usage from `similarity.py:4` and `graph_builder.py:8`:
<syntaxhighlight lang="python">
import networkx as nx

# GED calculation using NetworkX
for cost, (node_edit_path, edge_edit_path) in nx.optimize_edit_paths(
    g1, g2,
    node_ins_cost=node_ins_cost,
    node_del_cost=node_del_cost,
    node_subst_cost=node_subst_cost,
    edge_ins_cost=edge_ins_cost,
    edge_del_cost=edge_del_cost,
    edge_subst_cost=edge_subst_cost,
):
</syntaxhighlight>

YAML config loading from `config_loader.py:251-255`:
<syntaxhighlight lang="python">
@classmethod
def from_yaml(cls, path: Path) -> "WorkflowComparisonConfig":
    """Load configuration from YAML file"""
    with open(path) as f:
        data = yaml.safe_load(f)
    return cls._from_dict(data)
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `Config file not found: {path}` || Invalid config path || Check that config file exists at specified path
|-
|| `Preset '{name}' not found` || Invalid preset name || Use valid preset: `strict`, `standard`, or `lenient`
|-
|| `Unsupported config format: {suffix}` || Invalid config file type || Use `.yaml`, `.yml`, or `.json` format
|-
|| `ModuleNotFoundError: No module named 'networkx'` || Missing dependency || `pip install networkx>=3.2`
|}

== Compatibility Notes ==

* **Cross-platform:** Works on Windows, Linux, and macOS.
* **Memory Usage:** GED algorithm complexity grows with workflow size. Large workflows (50+ nodes) may require significant memory.
* **Config Formats:** Supports YAML (`.yaml`, `.yml`) and JSON (`.json`) configuration files.
* **Presets:** Built-in presets available: `strict`, `standard`, `lenient`.

== Related Pages ==

* [[requires_env::Implementation:n8n-io_n8n_load_workflow]]
* [[requires_env::Implementation:n8n-io_n8n_load_config]]
* [[requires_env::Implementation:n8n-io_n8n_build_workflow_graph]]
* [[requires_env::Implementation:n8n-io_n8n_relabel_graph_by_structure]]
* [[requires_env::Implementation:n8n-io_n8n_calculate_graph_edit_distance]]
* [[requires_env::Implementation:n8n-io_n8n_extract_operations_from_path]]
* [[requires_env::Implementation:n8n-io_n8n_similarity_formula]]
* [[requires_env::Implementation:n8n-io_n8n_format_output]]
