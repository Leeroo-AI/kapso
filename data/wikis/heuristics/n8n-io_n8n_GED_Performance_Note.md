# Heuristic: GED Performance for Small Graphs

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n-io/n8n|https://github.com/n8n-io/n8n]]
* [[source::Code|similarity.py|packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/similarity.py]]
|-
! Domains
| [[domain::Graph_Algorithms]], [[domain::Performance]], [[domain::AI_Evaluation]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==
Graph Edit Distance (GED) is computationally expensive but acceptable for workflow graphs because they are typically small (< 50 nodes).

=== Description ===
The GED algorithm has exponential worst-case complexity, making it impractical for large graphs. However, n8n workflows are constrained in size - most have fewer than 20 nodes. The code explicitly notes this constraint, enabling the use of exact GED calculation (`upper_bound=None`) rather than approximations.

=== Usage ===
When using the workflow comparison tool:
* For typical workflows (< 50 nodes): Exact GED works well
* For very large workflows: Expect slower performance
* If GED fails: A fallback basic calculation is automatically used

== The Insight (Rule of Thumb) ==

* **Action:** Use `nx.optimize_edit_paths()` with `upper_bound=None` for exact calculation
* **Value:** Works well for graphs under ~50 nodes (typical n8n workflow size)
* **Trade-off:** Exact results vs. computation time on large graphs
* **Fallback:** Basic edit cost calculation if GED algorithm fails

== Reasoning ==

1. **GED complexity:** O(n!) in worst case - unusable for large graphs

2. **Workflow constraints:** n8n workflows rarely exceed 50 nodes, making exact GED feasible

3. **Accuracy priority:** For AI evaluation, approximate similarity scores would reduce confidence in results

4. **Defensive programming:** Fallback calculation ensures the tool never fails completely

5. **User warning:** If GED fails, a warning is printed to inform users of reduced accuracy

== Code Evidence ==

From `similarity.py:80-81`:

<syntaxhighlight lang="python">
# Calculate GED using NetworkX
# Note: This can be slow for large graphs, but workflow graphs are typically small
</syntaxhighlight>

From `similarity.py:85-93`:

<syntaxhighlight lang="python">
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

Fallback handling from `similarity.py:123-127`:

<syntaxhighlight lang="python">
except Exception as e:
    # Fallback if NetworkX GED fails
    print(f"Warning: GED calculation failed, using fallback: {e}")
    edit_cost = _calculate_basic_edit_cost(g1, g2, config)
    edit_ops = []
</syntaxhighlight>

== Related Pages ==

* [[uses_heuristic::Implementation:n8n-io_n8n_calculate_graph_edit_distance]]
* [[uses_heuristic::Workflow:n8n-io_n8n_AI_Workflow_Comparison]]
* [[uses_heuristic::Principle:n8n-io_n8n_Graph_Edit_Distance]]
