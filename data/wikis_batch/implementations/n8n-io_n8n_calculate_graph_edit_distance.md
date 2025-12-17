{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Graph_Algorithms]], [[domain::Similarity_Metrics]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Concrete tool for calculating graph edit distance with custom cost functions between two workflow graphs, provided by the n8n workflow comparison system.

=== Description ===

The `calculate_graph_edit_distance` function computes the structural similarity between two n8n workflow graphs using graph edit distance (GED). It:
* Relabels graphs by structure to focus on topology rather than names
* Applies custom cost functions for node and edge operations
* Calculates normalized similarity score (0.0 to 1.0)
* Extracts detailed edit operations showing differences
* Provides comprehensive comparison metadata

This is the core comparison engine that determines how similar two workflows are structurally.

=== Usage ===

Use this implementation when you need to:
* Compare two workflow graphs for structural similarity
* Calculate precise edit distance between workflows
* Generate detailed diff operations between workflows
* Normalize similarity scores for threshold-based decisions

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/similarity.py
* '''Lines:''' L19-144

=== Signature ===
<syntaxhighlight lang="python">
def calculate_graph_edit_distance(
    g1: nx.DiGraph,
    g2: nx.DiGraph,
    config: WorkflowComparisonConfig
) -> Dict[str, Any]:
    """Calculate graph edit distance with custom cost functions.

    Args:
        g1: First workflow graph
        g2: Second workflow graph
        config: Configuration with cost functions and comparison settings

    Returns:
        Dictionary containing:
        - similarity_score: Normalized similarity (0.0 to 1.0)
        - edit_cost: Raw edit distance cost
        - max_cost: Maximum possible cost
        - operations: List of edit operations
        - metadata: Comparison metadata
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.similarity import calculate_graph_edit_distance
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| g1 || nx.DiGraph || Yes || First workflow graph (reference/expected)
|-
| g2 || nx.DiGraph || Yes || Second workflow graph (candidate/actual)
|-
| config || WorkflowComparisonConfig || Yes || Configuration with cost functions and max_cost
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| similarity_score || float || Normalized similarity score between 0.0 and 1.0 (1.0 = identical)
|-
| edit_cost || float || Raw graph edit distance cost
|-
| max_cost || float || Maximum possible cost (from config)
|-
| operations || List[Dict] || Detailed list of edit operations (insertions, deletions, substitutions)
|-
| node_counts || Dict || Node count statistics for both graphs
|-
| edge_counts || Dict || Edge count statistics for both graphs
|}

=== Operation Format ===
Each operation in the operations list contains:
{| class="wikitable"
|-
! Field !! Type !! Description
|-
| type || str || Operation type: "node_insert", "node_delete", "node_substitute", "edge_insert", "edge_delete"
|-
| description || str || Human-readable description of the operation
|-
| cost || float || Cost contribution of this operation
|-
| details || Dict || Additional context (node types, edge info, etc.)
|}

== Usage Examples ==

=== Basic Similarity Calculation ===
<syntaxhighlight lang="python">
from src.config_loader import load_config
from src.graph_builder import build_workflow_graph
from src.compare_workflows import load_workflow
from src.similarity import calculate_graph_edit_distance

# Load configuration
config = load_config("preset:balanced")

# Load and build graphs
workflow1 = load_workflow("workflows/version1.json")
workflow2 = load_workflow("workflows/version2.json")
graph1 = build_workflow_graph(workflow1, config)
graph2 = build_workflow_graph(workflow2, config)

# Calculate similarity
result = calculate_graph_edit_distance(graph1, graph2, config)

print(f"Similarity Score: {result['similarity_score']:.2%}")
print(f"Edit Cost: {result['edit_cost']}")
print(f"Max Cost: {result['max_cost']}")
</syntaxhighlight>

=== Analyzing Edit Operations ===
<syntaxhighlight lang="python">
# Calculate distance
result = calculate_graph_edit_distance(graph1, graph2, config)

# Analyze operations
print(f"Total operations: {len(result['operations'])}")

# Group by operation type
from collections import Counter
op_types = Counter(op['type'] for op in result['operations'])
print(f"Node insertions: {op_types['node_insert']}")
print(f"Node deletions: {op_types['node_delete']}")
print(f"Node substitutions: {op_types['node_substitute']}")
print(f"Edge insertions: {op_types['edge_insert']}")
print(f"Edge deletions: {op_types['edge_delete']}")

# Show detailed operations
print("\nDetailed Operations:")
for op in result['operations']:
    print(f"  [{op['type']:20s}] {op['description']:50s} (cost: {op['cost']})")
</syntaxhighlight>

=== Threshold-Based Comparison ===
<syntaxhighlight lang="python">
def workflows_are_similar(graph1, graph2, config, threshold=0.8):
    """Check if workflows meet similarity threshold."""
    result = calculate_graph_edit_distance(graph1, graph2, config)
    return result['similarity_score'] >= threshold

# Compare multiple workflow pairs
workflow_pairs = [
    ("workflows/v1.json", "workflows/v2.json"),
    ("workflows/prod.json", "workflows/staging.json"),
    ("workflows/original.json", "workflows/modified.json"),
]

config = load_config("preset:strict")

for wf1_path, wf2_path in workflow_pairs:
    wf1 = load_workflow(wf1_path)
    wf2 = load_workflow(wf2_path)
    g1 = build_workflow_graph(wf1, config)
    g2 = build_workflow_graph(wf2, config)

    is_similar = workflows_are_similar(g1, g2, config, threshold=0.85)
    result = calculate_graph_edit_distance(g1, g2, config)

    status = "PASS" if is_similar else "FAIL"
    print(f"{status}: {wf1_path} vs {wf2_path} - {result['similarity_score']:.2%}")
</syntaxhighlight>

=== Comparing with Different Configurations ===
<syntaxhighlight lang="python">
# Compare using multiple presets
presets = ["strict", "balanced", "lenient"]

for preset_name in presets:
    config = load_config(f"preset:{preset_name}")
    result = calculate_graph_edit_distance(graph1, graph2, config)

    print(f"\n{preset_name.upper()} Configuration:")
    print(f"  Similarity: {result['similarity_score']:.2%}")
    print(f"  Edit Cost: {result['edit_cost']}")
    print(f"  Operations: {len(result['operations'])}")
</syntaxhighlight>

=== Generating Comparison Report ===
<syntaxhighlight lang="python">
import json

def generate_comparison_report(graph1, graph2, config, output_file):
    """Generate detailed comparison report."""
    result = calculate_graph_edit_distance(graph1, graph2, config)

    # Enrich with additional statistics
    report = {
        "summary": {
            "similarity_score": result['similarity_score'],
            "similarity_percentage": f"{result['similarity_score'] * 100:.1f}%",
            "edit_cost": result['edit_cost'],
            "max_cost": result['max_cost'],
        },
        "graph_statistics": {
            "graph1": {
                "nodes": result['node_counts']['g1'],
                "edges": result['edge_counts']['g1'],
            },
            "graph2": {
                "nodes": result['node_counts']['g2'],
                "edges": result['edge_counts']['g2'],
            },
        },
        "operations": result['operations'],
        "comparison_verdict": "PASS" if result['similarity_score'] >= 0.8 else "FAIL",
    }

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    return report

# Generate report
report = generate_comparison_report(
    graph1,
    graph2,
    config,
    "comparison_report.json"
)

print(f"Comparison verdict: {report['comparison_verdict']}")
</syntaxhighlight>

=== Batch Workflow Comparison ===
<syntaxhighlight lang="python">
import os
from pathlib import Path

def compare_workflow_directory(reference_path, candidate_dir, config):
    """Compare reference workflow against all candidates in directory."""
    reference = load_workflow(reference_path)
    ref_graph = build_workflow_graph(reference, config)

    results = []

    for candidate_file in Path(candidate_dir).glob("*.json"):
        candidate = load_workflow(str(candidate_file))
        cand_graph = build_workflow_graph(candidate, config)

        result = calculate_graph_edit_distance(ref_graph, cand_graph, config)
        results.append({
            "file": candidate_file.name,
            "similarity": result['similarity_score'],
            "operations": len(result['operations']),
        })

    # Sort by similarity descending
    results.sort(key=lambda x: x['similarity'], reverse=True)

    print(f"Comparison against reference: {reference_path}\n")
    for r in results:
        print(f"  {r['file']:30s} - {r['similarity']:.2%} ({r['operations']} ops)")

    return results

# Compare reference against candidates
config = load_config("preset:balanced")
results = compare_workflow_directory(
    "workflows/reference.json",
    "workflows/candidates/",
    config
)
</syntaxhighlight>

== Implementation Details ==

=== Graph Relabeling ===
The function first relabels both graphs using structural identifiers via `_relabel_graph_by_structure()`. This ensures comparison is based on structure, not node names.

=== Edit Path Optimization ===
Uses NetworkX's `optimize_edit_paths()` to find the optimal sequence of edit operations:
<syntaxhighlight lang="python">
edit_path_generator = nx.optimize_edit_paths(
    g1_relabeled,
    g2_relabeled,
    node_match=node_match_fn,
    edge_match=edge_match_fn,
    node_subst_cost=node_subst_cost_fn,
    node_del_cost=node_del_cost_fn,
    node_ins_cost=node_ins_cost_fn,
    edge_subst_cost=edge_subst_cost_fn,
    edge_del_cost=edge_del_cost_fn,
    edge_ins_cost=edge_ins_cost_fn,
)
</syntaxhighlight>

=== Similarity Score Calculation ===
The similarity score is normalized using:
<syntaxhighlight lang="python">
similarity_score = max(0.0, min(1.0, 1.0 - (edit_cost / max_cost)))
</syntaxhighlight>

This ensures:
* Score of 1.0 means identical workflows (zero edit cost)
* Score of 0.0 means maximally different (edit cost >= max_cost)
* Scores are clamped to [0.0, 1.0] range

=== Cost Functions ===
The function uses configuration-defined cost functions for:
* Node insertion/deletion/substitution
* Edge insertion/deletion/substitution
* Custom matching criteria for nodes and edges

== Performance Considerations ==

=== Time Complexity ===
Graph edit distance is NP-hard. The algorithm uses heuristics for optimization, but complexity grows with:
* Number of nodes (exponential in worst case)
* Number of edges
* Graph structure complexity

=== Optimization Strategies ===
* Takes only the first edit path (best match) to avoid exhaustive search
* Uses structural relabeling to reduce search space
* Applies filtering via configuration to reduce graph size

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_GED_Calculation]]

=== Uses ===
* [[uses::Implementation:n8n-io_n8n_relabel_graph_by_structure]]
* [[uses::Implementation:n8n-io_n8n_load_config]]
* [[uses::Implementation:n8n-io_n8n_similarity_formula]]

=== Used By ===
* [[used_by::Implementation:n8n-io_n8n_format_output]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Workflow_Comparison]]
* [[requires_dependency::networkx]]

[[Category:Implementation]]
[[Category:n8n]]
[[Category:Graph_Algorithms]]
