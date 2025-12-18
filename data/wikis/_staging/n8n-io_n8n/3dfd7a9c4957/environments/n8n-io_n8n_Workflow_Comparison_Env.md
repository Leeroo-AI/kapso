# Environment: Workflow Comparison Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n-io/n8n|https://github.com/n8n-io/n8n]]
* [[source::Doc|pyproject.toml|packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/pyproject.toml]]
|-
! Domains
| [[domain::AI_Evaluation]], [[domain::Graph_Algorithms]], [[domain::Workflow_Analysis]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==
Python 3.11+ environment with NetworkX, NumPy, and SciPy for graph-based workflow similarity comparison using graph edit distance algorithms.

=== Description ===
This environment provides the runtime context for the n8n AI Workflow Comparison tool, used to evaluate AI-generated workflows against ground truth by computing graph edit distance (GED). It relies heavily on NetworkX's graph algorithms (`nx.optimize_edit_paths`) and scientific computing libraries for numerical operations.

=== Usage ===
Use this environment when running the **AI Workflow Comparison** workflow for evaluating AI-generated n8n workflows. It is required for all implementations in the comparison pipeline including `load_config`, `build_workflow_graph`, `calculate_graph_edit_distance`, and related functions.

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux, macOS, or Windows || No platform-specific restrictions
|-
| Python || >= 3.11 || Type hints and stdlib features
|-
| Memory || Varies with graph size || GED is computationally intensive for large graphs
|}

== Dependencies ==

=== System Packages ===
* Python 3.11+

=== Python Packages ===
* `networkx` >= 3.2 - Graph data structures and algorithms
* `numpy` >= 2.3.4 - Numerical computing
* `pyyaml` >= 6.0 - YAML configuration parsing
* `scipy` >= 1.16.3 - Scientific computing (used by NetworkX GED)

== Quick Install ==

<syntaxhighlight lang="bash">
# Requires Python 3.11+
pip install networkx>=3.2 numpy>=2.3.4 pyyaml>=6.0 scipy>=1.16.3
</syntaxhighlight>

== Credentials ==

No credentials are required. This is a standalone evaluation tool that processes local workflow JSON files.

== Code Evidence ==

Dependencies from `pyproject.toml:6-11`:
<syntaxhighlight lang="toml">
dependencies = [
    "networkx>=3.2",
    "numpy>=2.3.4",
    "pyyaml>=6.0",
    "scipy>=1.16.3",
]
</syntaxhighlight>

Python version requirement from `pyproject.toml:5`:
<syntaxhighlight lang="toml">
requires-python = ">=3.11"
</syntaxhighlight>

NetworkX GED usage from `similarity.py:85-93`:
<syntaxhighlight lang="python">
# Use optimize_edit_paths with edge_match instead of edge cost functions
# This prevents false positive edge insertions/deletions
edit_path_generator = nx.optimize_edit_paths(
    g1_relabeled,
    g2_relabeled,
    node_subst_cost=node_subst_cost,
    node_del_cost=node_del_cost,
    node_ins_cost=node_ins_cost,
    edge_match=edge_match,
    upper_bound=None,  # Calculate exact
)
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `ModuleNotFoundError: No module named 'networkx'` || NetworkX not installed || `pip install networkx>=3.2`
|-
|| `ModuleNotFoundError: No module named 'numpy'` || NumPy not installed || `pip install numpy>=2.3.4`
|-
|| `ModuleNotFoundError: No module named 'yaml'` || PyYAML not installed || `pip install pyyaml>=6.0`
|-
|| `Warning: GED calculation failed, using fallback` || GED algorithm failed on graphs || Normal fallback behavior for problematic graphs
|-
|| `FileNotFoundError` for workflow file || Workflow JSON file not found || Verify file path is correct
|-
|| `KeyError: 'nodes'` || Invalid workflow JSON format || Ensure workflow JSON has `nodes` and `connections` keys
|}

== Compatibility Notes ==

* **Performance:** The GED algorithm can be slow for large graphs. The code notes: "This can be slow for large graphs, but workflow graphs are typically small."
* **Fallback:** If NetworkX GED fails, a basic edit cost calculation is used as fallback.
* **Configuration:** Supports YAML and JSON configuration files, plus built-in presets (e.g., `preset:strict`).

== Related Pages ==

* [[requires_env::Implementation:n8n-io_n8n_load_config]]
* [[requires_env::Implementation:n8n-io_n8n_load_workflow]]
* [[requires_env::Implementation:n8n-io_n8n_build_workflow_graph]]
* [[requires_env::Implementation:n8n-io_n8n_relabel_graph_by_structure]]
* [[requires_env::Implementation:n8n-io_n8n_calculate_graph_edit_distance]]
* [[requires_env::Implementation:n8n-io_n8n_extract_operations_from_path]]
* [[requires_env::Implementation:n8n-io_n8n_calculate_max_cost]]
* [[requires_env::Implementation:n8n-io_n8n_determine_priority]]
* [[requires_env::Implementation:n8n-io_n8n_format_output]]
