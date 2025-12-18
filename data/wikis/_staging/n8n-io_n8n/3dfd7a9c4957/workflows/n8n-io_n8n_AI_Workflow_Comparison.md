# AI Workflow Comparison

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
* [[source::Doc|NetworkX|https://networkx.org/documentation/stable/]]
|-
! Domains
| [[domain::AI_Evaluation]], [[domain::Graph_Algorithms]], [[domain::Workflow_Automation]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:30 GMT]]
|}

== Overview ==

End-to-end process for evaluating AI-generated n8n workflows against ground truth workflows using graph edit distance and configurable cost functions.

=== Description ===

This workflow documents n8n's system for programmatically evaluating AI-generated workflows:

1. **Goal:** Produce a similarity score (0-1) and actionable edit list showing how to transform a generated workflow to match the expected ground truth.
2. **Scope:** Covers JSON workflow parsing, graph construction, configurable filtering, graph edit distance calculation, and human-readable output formatting.
3. **Strategy:** Converts workflows to NetworkX directed graphs, then uses optimal graph edit distance with custom cost functions to measure structural and semantic similarity.

The comparison system supports multiple configuration presets (strict, standard, lenient) and allows node/parameter filtering to focus on semantically significant differences.

=== Usage ===

Execute this workflow when:
- Evaluating AI workflow builder output quality
- Benchmarking different AI models or prompts for workflow generation
- Regression testing workflow generation after model updates
- Understanding specific differences between generated and expected workflows
- Fine-tuning cost weights for domain-specific workflow types

== Execution Steps ==

=== Step 1: Configuration Loading ===
[[step::Principle:n8n-io_n8n_Comparison_Configuration]]

Load comparison configuration from YAML/JSON file or built-in preset. The configuration defines cost weights, ignore rules, similarity groups, and output formatting options.

'''Key considerations:'''
* Presets: strict (high sensitivity), standard (balanced), lenient (forgiving)
* Cost weights for node insertion/deletion/substitution
* Similarity groups define equivalent node types (e.g., HTTP variants)
* Ignore rules filter out cosmetic differences (metadata, positioning)
* Parameter rules can use fuzzy/semantic/numeric comparison

=== Step 2: Workflow JSON Parsing ===
[[step::Principle:n8n-io_n8n_Workflow_Parsing]]

Load and parse n8n workflow JSON files for both the generated and ground truth workflows. The parser extracts the nodes array and connections object that define the workflow structure.

'''What happens:'''
* Read JSON files from filesystem
* Extract nodes with name, type, typeVersion, parameters
* Extract connections mapping source nodes to targets
* Handle missing or malformed JSON gracefully
* Return structured workflow dictionaries

=== Step 3: Graph Construction ===
[[step::Principle:n8n-io_n8n_Graph_Construction]]

Convert n8n workflow JSON structures to NetworkX directed graphs. Each node becomes a graph vertex with attributes, and connections become directed edges with metadata.

'''Pseudocode:'''
  1. Create empty DiGraph
  2. For each workflow node (if not ignored by config):
     - Filter parameters based on ignore rules
     - Detect if node is trigger (by type/name patterns)
     - Add node with type, typeVersion, parameters, is_trigger
  3. For each connection:
     - Skip if source/target node was filtered
     - Skip if connection type is ignored
     - Add edge with connection_type, indices, node types

=== Step 4: Graph Relabeling ===
[[step::Principle:n8n-io_n8n_Graph_Relabeling]]

Relabel graph nodes using structural IDs instead of display names to ensure nodes are matched by type and position rather than naming conventions.

'''What happens:'''
* Separate nodes into triggers and non-triggers
* Sort by type, out-degree, in-degree, name
* Assign structural IDs: trigger_0, trigger_1, node_0, node_1, etc.
* Preserve original name in _original_name attribute
* Store normalized name hash for smart quote handling

=== Step 5: Graph Edit Distance Calculation ===
[[step::Principle:n8n-io_n8n_Graph_Edit_Distance]]

Calculate the optimal graph edit distance between the two workflow graphs using NetworkX's optimize_edit_paths algorithm with custom cost functions.

'''Key considerations:'''
* Node substitution cost varies by type match (same, similar, different, trigger)
* Parameter mismatches add weighted cost to substitution
* Edge matching uses connection type equivalence groups
* Falls back to basic cost calculation if GED fails
* Returns edit path with node and edge operations

=== Step 6: Edit Operation Extraction ===
[[step::Principle:n8n-io_n8n_Edit_Operation_Extraction]]

Extract and categorize edit operations from the GED algorithm's edit path. Each operation is assigned a type, description, cost, and priority level.

'''Operation types:'''
* node_insert: Add missing node (with type info)
* node_delete: Remove extra node (with type info)
* node_substitute: Change node type or parameters
* edge_insert: Add missing connection
* edge_delete: Remove extra connection
* edge_substitute: Modify connection type

=== Step 7: Similarity Score Computation ===
[[step::Principle:n8n-io_n8n_Similarity_Scoring]]

Compute the final similarity score by normalizing the edit cost against the theoretical maximum cost. Score ranges from 0 (completely different) to 1 (identical).

'''Pseudocode:'''
  1. Calculate max_cost = cost to delete g1 + insert g2
  2. similarity = 1 - (edit_cost / max_cost)
  3. Clamp to [0, 1] range
  4. Handle edge cases: empty graphs = 1.0, zero max = depends on edit_cost

=== Step 8: Priority Assignment ===
[[step::Principle:n8n-io_n8n_Priority_Assignment]]

Assign priority levels (critical, major, minor) to each edit operation based on cost thresholds and node importance (trigger status).

'''Priority rules:'''
* Critical: Trigger insert/delete, or cost >= 80% of trigger mismatch cost
* Major: Cost >= 80% of different_type substitution cost
* Minor: All other operations

=== Step 9: Output Formatting ===
[[step::Principle:n8n-io_n8n_Output_Formatting]]

Format the comparison results for output in JSON or human-readable summary format. Include metadata about graph statistics and configuration.

'''Output includes:'''
* similarity_score and percentage
* edit_cost and max_possible_cost
* top_edits list sorted by cost
* metadata: node counts, config name, filter statistics
* Summary format: PASS/FAIL indicator at 70% threshold

== Execution Diagram ==

{{#mermaid:graph TD
    A[Configuration Loading] --> B[Workflow JSON Parsing]
    B --> C[Graph Construction]
    C --> D[Graph Relabeling]
    D --> E[Graph Edit Distance Calculation]
    E --> F[Edit Operation Extraction]
    F --> G[Similarity Score Computation]
    G --> H[Priority Assignment]
    H --> I[Output Formatting]
}}

== Related Pages ==

=== Steps ===
* [[step::Principle:n8n-io_n8n_Comparison_Configuration]]
* [[step::Principle:n8n-io_n8n_Workflow_Parsing]]
* [[step::Principle:n8n-io_n8n_Graph_Construction]]
* [[step::Principle:n8n-io_n8n_Graph_Relabeling]]
* [[step::Principle:n8n-io_n8n_Graph_Edit_Distance]]
* [[step::Principle:n8n-io_n8n_Edit_Operation_Extraction]]
* [[step::Principle:n8n-io_n8n_Similarity_Scoring]]
* [[step::Principle:n8n-io_n8n_Priority_Assignment]]
* [[step::Principle:n8n-io_n8n_Output_Formatting]]

=== Related Concepts ===
* [[related::Heuristic:n8n-io_n8n_GED_Performance_Note]] - GED algorithm performance considerations
