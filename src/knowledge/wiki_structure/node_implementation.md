# Node Type: Implementation

## Definition

**Role:** Tool (Source of Truth for Syntax)

An **Implementation** represents concrete code - a class, function, or module. It serves as the authoritative reference for how to use specific functionality in the repository.

## Purpose

- Provides the "Source of Truth" for syntax
- Documents public APIs users actually import
- Maps I/O contracts

## Key Characteristics

| Aspect | Description |
|:---|:---|
| Type | Tool |
| Function | Source of Truth for Syntax |
| Scope | Concrete code (class, function, module) |

## Template Structure

```mediawiki
= Implementation: {Library.ClassName} =
[[Category:Implementations]]

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|{Repo_Name}|{URL}]]
* [[source::Doc|{Doc_Title}|{URL}]]
|-
! Domains
| [[domain::Domain_Tag_1]], [[domain::Domain_Tag_2]]
|-
! Last Updated
| [[last_updated::YYYY-MM-DD HH:MM GMT]]
|}

== Overview ==
{Brief description: What this specific class/function does.}

== Code Signature ==
<syntaxhighlight lang="python">
class ClassName:
    ...
</syntaxhighlight>

== I/O Contract ==
* **Consumes:** {Input_Config_Schema}
* **Produces:** {Output_Model_Weights}

== Related Pages ==
=== Context & Requirements ===
* [[requires_env::Environment:{Env_Name}]] - Environment setup

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:{Heuristic_Name}]] - Hacks/Optimizations
```

## Semantic Links

| Link Type | Target | Description |
|:---|:---|:---|
| `requires_env::Environment` | Environment | Required environment setup |
| `uses_heuristic::Heuristic` | Heuristic | Specific hacks/optimizations for this tool |
