{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
* [[source::Doc|n8n Docs|https://docs.n8n.io]]
|-
! Domains
| [[domain::AI_Evaluation]], [[domain::Graph_Algorithms]], [[domain::Testing]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

Process for evaluating AI-generated n8n workflows against ground truth using graph edit distance algorithms to calculate structural similarity.

=== Description ===

This workflow describes the programmatic evaluation system for the AI Workflow Builder. It converts n8n workflow JSON structures into NetworkX directed graphs, then calculates graph edit distance (GED) to measure similarity. The system uses configurable cost functions with presets (strict/standard/lenient) to weight different types of graph edits differently, prioritizing trigger node matches and handling parameter differences intelligently.

=== Usage ===

Execute this workflow when:
* Evaluating AI-generated workflows against human-created ground truth
* Testing the AI Workflow Builder's output quality
* Performing batch evaluation of multiple workflow pairs
* Integrating workflow comparison into CI/CD pipelines

The CLI tool (`compare_workflows.py`) provides both JSON and human-readable output formats.

== Execution Steps ==

=== Step 1: Workflow Loading ===
[[step::Principle:n8n-io_n8n_Workflow_Loading]]

Load both the generated workflow and ground truth workflow from JSON files. The workflows are n8n-format JSON with "nodes" and "connections" arrays defining the workflow structure.

'''Input validation:'''
* File existence checked with helpful error messages
* JSON parsing with decode error reporting
* Both workflows loaded before any processing begins

=== Step 2: Configuration Loading ===
[[step::Principle:n8n-io_n8n_Configuration_Loading]]

Load comparison configuration from a preset, custom YAML/JSON file, or defaults. Configuration controls cost weights, ignored parameters, equivalent connection types, and filtering rules.

'''Configuration options:'''
* Presets: strict (high costs, few ignores), standard (balanced), lenient (low costs, many ignores)
* Custom files allow fine-tuned parameter ignore rules per node type
* Configuration includes node/edge insertion/deletion/substitution costs
* Expression patterns can normalize $fromAI() calls

=== Step 3: Graph Construction ===
[[step::Principle:n8n-io_n8n_Graph_Construction]]

Convert each workflow JSON into a NetworkX DiGraph. Nodes become graph vertices with attributes (type, typeVersion, parameters, is_trigger). Connections become directed edges with connection metadata.

'''Graph building process:'''
* Nodes filtered based on ignore rules (e.g., sticky notes)
* Parameters filtered per node type (e.g., ignore position, notesInFlow)
* Trigger nodes detected by type/name patterns
* Edges include connection_type, source_index, target_index

=== Step 4: Graph Relabeling ===
[[step::Principle:n8n-io_n8n_Graph_Relabeling]]

Relabel graph nodes using structural IDs instead of display names. This ensures GED matches nodes by type and position rather than user-assigned names. Triggers get "trigger_N" labels, others get "node_N".

'''Relabeling benefits:'''
* Prevents false mismatches from different node names
* Preserves original names as _original_name attribute
* Adds _name_hash for smart quote normalization
* Deterministic ordering for consistent results

=== Step 5: GED Calculation ===
[[step::Principle:n8n-io_n8n_GED_Calculation]]

Calculate graph edit distance using NetworkX's optimize_edit_paths with custom cost functions. Node operations (insert/delete/substitute) use type-aware costs. Edge matching uses equivalence checking rather than costs.

'''Cost function logic:'''
* Trigger operations cost more than regular nodes
* Same-type substitutions compare parameters for partial costs
* Different-type substitutions cost maximum
* Edge matching considers equivalent connection types (e.g., main ≡ ai_tool)

=== Step 6: Edit Operation Extraction ===
[[step::Principle:n8n-io_n8n_Edit_Extraction]]

Extract and describe the edit operations from the optimal edit path. Each operation gets a human-readable description, cost value, and priority level (critical/major/minor).

'''Operation types:'''
* node_insert: "Add missing node 'X' (type: Y)"
* node_delete: "Remove node 'X' (type: Y)"
* node_substitute: "Change node 'X' from type 'A' to 'B'" or "Update parameters of node 'X'"
* edge_insert/delete/substitute: Connection changes

=== Step 7: Similarity Calculation ===
[[step::Principle:n8n-io_n8n_Similarity_Calculation]]

Calculate the final similarity score as 1 - (edit_cost / max_possible_cost). Maximum cost is the theoretical cost of deleting all of g1 and inserting all of g2.

'''Score interpretation:'''
* 1.0 = identical workflows
* 0.0 = completely different
* >0.7 typically considered passing in evaluation
* Score clamped to [0, 1] range

=== Step 8: Result Formatting ===
[[step::Principle:n8n-io_n8n_Result_Formatting]]

Format comparison results for output. JSON format includes similarity_score, edit_cost, max_possible_cost, top_edits, and metadata. Summary format provides human-readable output with priority indicators.

'''Output formats:'''
* JSON: Machine-parseable with full details
* Summary: Visual with pass/fail indicator and priority emojis
* Verbose mode adds parameter diffs and graph statistics
* Exit code always 0; caller interprets similarity score

== Execution Diagram ==

{{#mermaid:graph TD
    A[Load Generated Workflow] --> C[Load Configuration]
    B[Load Ground Truth Workflow] --> C
    C --> D[Build Generated Graph]
    C --> E[Build Ground Truth Graph]
    D --> F[Relabel by Structure]
    E --> G[Relabel by Structure]
    F --> H[Calculate GED]
    G --> H
    H --> I[Extract Edit Operations]
    I --> J[Calculate Similarity Score]
    J --> K[Calculate Max Cost]
    K --> L{Output Format?}
    L -->|JSON| M[Format as JSON]
    L -->|Summary| N[Format as Summary]
    M --> O[Output Result]
    N --> O
}}

== Related Pages ==

* [[step::Principle:n8n-io_n8n_Workflow_Loading]]
* [[step::Principle:n8n-io_n8n_Configuration_Loading]]
* [[step::Principle:n8n-io_n8n_Graph_Construction]]
* [[step::Principle:n8n-io_n8n_Graph_Relabeling]]
* [[step::Principle:n8n-io_n8n_GED_Calculation]]
* [[step::Principle:n8n-io_n8n_Edit_Extraction]]
* [[step::Principle:n8n-io_n8n_Similarity_Calculation]]
* [[step::Principle:n8n-io_n8n_Result_Formatting]]
