{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|YAML Specification|https://yaml.org]]
* [[source::Doc|n8n Workflow Comparison Configuration|https://docs.n8n.io]]
|-
! Domains
| [[domain::Configuration_Management]], [[domain::Workflow_Analysis]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Configuration Loading is the principle of importing comparison settings from multiple sources including built-in presets, custom YAML/JSON files, and sensible defaults to control how workflows are compared.

=== Description ===

The configuration loading principle provides a flexible, hierarchical system for defining how workflow comparisons should be performed. It supports three levels of configuration:

1. **Built-in Presets**: Pre-defined configurations (strict, standard, lenient) that cover common comparison scenarios
2. **Custom Configuration Files**: User-provided YAML or JSON files that specify custom cost weights and ignore rules
3. **Default Fallbacks**: Sensible defaults used when no configuration is specified

Configuration controls two main aspects:
* **Cost Weights**: How expensive different edit operations are (node insertion, deletion, parameter changes)
* **Ignore Rules**: Which differences should be disregarded (timestamps, UI positions, credentials)

This layered approach allows users to start with presets and customize as needed without overwhelming them with complexity.

=== Usage ===

Apply this principle when:
* Building configurable comparison tools that need to accommodate different use cases
* Allowing users to define custom similarity metrics
* Implementing preset configurations for common scenarios
* Creating tools that balance simplicity with flexibility

== Theoretical Basis ==

=== Configuration Hierarchy ===

```
Priority: Custom Config > Preset > Defaults
```

=== Preset Types ===

1. **Strict Mode**:
   - High costs for all changes
   - Minimal ignore rules
   - Use case: Exact workflow matching

2. **Standard Mode**:
   - Balanced costs
   - Ignore metadata and UI positions
   - Use case: Functional equivalence checking

3. **Lenient Mode**:
   - Lower costs for parameter changes
   - Extensive ignore rules
   - Use case: Structural similarity detection

=== Configuration Schema ===

```yaml
cost_weights:
  node_insertion: 1.0
  node_deletion: 1.0
  node_substitution: 0.5
  edge_insertion: 0.5
  edge_deletion: 0.5
  parameter_change: 0.1

ignore_rules:
  - field: "position"
  - field: "id"
  - field_pattern: ".*_timestamp$"
```

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_load_config]]

=== Related Principles ===
* [[related::Principle:n8n-io_n8n_GED_Calculation]]
* [[related::Principle:n8n-io_n8n_Similarity_Calculation]]
