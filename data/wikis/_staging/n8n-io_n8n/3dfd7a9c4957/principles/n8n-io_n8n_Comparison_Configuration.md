# Principle: Comparison Configuration

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

Principle for loading and structuring workflow comparison configuration from multiple sources including presets, YAML, and JSON files.

=== Description ===

Comparison Configuration manages the policy settings for workflow evaluation:

1. **Cost Weights**: Configurable costs for node/edge insertion, deletion, and substitution
2. **Similarity Groups**: Node types that are considered equivalent (e.g., HTTP nodes)
3. **Ignore Rules**: Parameters, node types, or patterns to exclude from comparison
4. **Exemptions**: Nodes that are optional in generated or ground truth
5. **Output Settings**: Maximum edits to report, grouping strategy

Configuration sources:
- **Default**: Built-in sensible defaults
- **Presets**: Named configurations (strict, standard, lenient)
- **Custom Files**: User-provided YAML or JSON configuration

=== Usage ===

Apply this principle when:
- Building configurable evaluation systems
- Implementing comparison tools with tunable parameters
- Creating test harnesses that need different strictness levels
- Designing ML evaluation pipelines with configurable metrics

== Theoretical Basis ==

Configuration loading follows a **Multi-Source Factory** pattern:

<syntaxhighlight lang="python">
# Pseudo-code for configuration loading

def load_config(config_source: Optional[str] = None) -> WorkflowComparisonConfig:
    # 1. No source - use defaults
    if not config_source:
        return WorkflowComparisonConfig()

    # 2. Preset source - load from built-in presets
    if config_source.startswith("preset:"):
        preset_name = config_source.split(":", 1)[1]
        return WorkflowComparisonConfig.load_preset(preset_name)

    # 3. File source - load from YAML/JSON
    path = Path(config_source)
    if path.suffix in [".yaml", ".yml"]:
        return WorkflowComparisonConfig.from_yaml(path)
    elif path.suffix == ".json":
        return WorkflowComparisonConfig.from_json(path)
</syntaxhighlight>

Configuration structure includes:
- **Node costs**: insertion, deletion, substitution (same/similar/different type)
- **Edge costs**: insertion, deletion, substitution
- **Parameter weights**: mismatch and nesting weights
- **Ignore rules**: node types, parameters, patterns

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_load_config]]
