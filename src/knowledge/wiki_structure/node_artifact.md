# Node Type: Artifact

## Definition

**Role:** Noun (Shape of Data)

An **Artifact** is a passive data object - a schema, config, or data structure. It defines the "shape" of data (JSON, YAML, CSV) to prevent hallucinated data structures.

## Purpose

- Defines exact data contracts
- Documents all keys, columns, and types
- Prevents "hallucinated data structures" in agents
- Ensures correct data is passed between functions

## Key Characteristics

| Aspect | Description |
|--------|-------------|
| Type | Noun |
| Function | Shape of Data |
| Scope | Passive data objects, schemas, configs |
| Nature | The Data Contract |

## Required Metadata

| Field | Description |
|-------|-------------|
| Repo URL | GitHub repository URL from `metadata.json` |
| Domain(s) | Up to 3 domain tags, comma-separated |
| Last Updated | Datetime in `YYYY-MM-DD HH:MM GMT` format |

## Template Structure

```mediawiki
= Artifact: {Name_of_Object} =
[[Category:Artifacts]]

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Repo URL
|| [{repoUrl} {repo_name}]
|-
! Domain(s)
|| {Domain_Tag_1}, {Domain_Tag_2}
|-
! Last Updated
|| {YYYY-MM-DD HH:MM GMT}
|}

== Description ==
{Brief description: e.g., "The configuration dictionary required to initialize the Trainer."}

== Schema Definition ==
{| class="wikitable"
! Key/Column !! Type !! Required !! Description
|-
|| learning_rate || float || Yes || Step size for optimization (e.g., 1e-4)
|-
|| batch_size || int || No || Samples per step. Default: 32.
|}

== Validation Example ==
<syntaxhighlight lang="json">
{
  "learning_rate": 0.0001,
  "batch_size": 64,
  "use_gpu": true
}
</syntaxhighlight>

== Code References ==
'''GitHub Repository:''' [{repoUrl} {repo_name}]

'''Schema/Config Files:'''
* [{repoUrl}/blob/{branch}/path/to/config.yaml config.yaml] - Configuration file

== Related Pages ==
* [[conforms_to::Principle:{repo_name}/{Related_Theory}]] - Theory this artifact conforms to
* [[consumed_by::Implementation:{repo_name}/{Consumer_Class}]] - Implementations consuming this
* [[produced_by::Implementation:{repo_name}/{Producer_Class}]] - Implementations producing this
```

## Coverage Target

**Minimum:** 5-8 pages per repository (including ALL workflow inputs/outputs)

## Sources to Scan

- `configs/` directory
- `schemas/` directory
- `data/` directory
- Type hints in code
- `__init__` arguments
- Request/response formats

## Critical Rule

**Never describe an input as just "a dictionary."**

You must define an Artifact with:
- Exact key names
- Data types
- Required vs optional fields
- Default values
- Example values

## Semantic Links

| Link Type | Target | Description |
|-----------|--------|-------------|
| `conforms_to::Principle` | Principle | Theoretical principle this conforms to |
| `consumed_by::Implementation` | Implementation | Code that consumes this artifact |
| `produced_by::Implementation` | Implementation | Code that produces this artifact |

## Schema Definition Best Practices

Always include in the schema table:
1. **Key/Column** - Exact name used in code
2. **Type** - Data type (string, int, float, bool, list, dict, etc.)
3. **Required** - Yes/No
4. **Description** - What it does, valid values, defaults

