# Node Type: Implementation

## Definition

**Role:** Tool (Source of Truth for Syntax)

An **Implementation** represents concrete code - a class, function, or module. It serves as the authoritative reference for how to use specific functionality in the repository.

## Purpose

- Provides the "Source of Truth" for syntax
- Documents public APIs users actually import
- Maps I/O contracts (what data goes in/out)
- Links code to theoretical Principles

## Key Characteristics

| Aspect | Description |
|--------|-------------|
| Type | Tool |
| Function | Source of Truth for Syntax |
| Scope | Concrete code (class, function, module) |
| Focus | Public API, not internal helpers |

## Required Metadata

| Field | Description |
|-------|-------------|
| Repo URL | GitHub repository URL from `metadata.json` |
| Domain(s) | Up to 3 domain tags, comma-separated |
| Last Updated | Datetime in `YYYY-MM-DD HH:MM GMT` format |

## Template Structure

```mediawiki
= Implementation: {Library.ClassName} =
[[Category:Implementations]]

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

== Overview ==
{Brief description: What this specific class/function does.}

== Code Signature ==
<syntaxhighlight lang="python">
class ClassName:
    def __init__(self, param1: int, config: dict):
        """
        Args:
            param1: ...
            config: ...
        """
        ...
</syntaxhighlight>

== I/O Contract ==
* **Consumes:** {Input_Config_Schema} - Description of input
* **Produces:** {Output_Model_Weights} - Description of output

== Usage Example ==
<syntaxhighlight lang="python">
from library import ClassName

# Initialize
tool = ClassName(param1=10, config={...})

# Execute
result = tool.run()
</syntaxhighlight>

== Code References ==
'''GitHub Repository:''' [{repoUrl} {repo_name}]

'''Source File:'''
* [{repoUrl}/blob/{branch}/path/to/file.py file.py] - Main implementation

== Related Pages ==
* [[implements::Principle:{repo_name}/{Theoretical_Principle}]] - Theory this implements
* [[requires_env::Environment:{repo_name}/{Env_Name}]] - Required environment
* [[consumes::Artifact:{repo_name}/{Input_Config_Schema}]] - Input artifact
* [[produces::Artifact:{repo_name}/{Output_Model_Weights}]] - Output artifact
* [[used_in::Workflow:{repo_name}/{Workflow_Name}]] - Workflow using this
```

## Coverage Target

**Minimum:** 6-10 pages per repository

## Sources to Scan

- `src/` directory
- `lib/` directory
- Main classes and public APIs
- Functions users actually import

## Scope Guidelines

- **Focus on Public API:** Document tools users import, not internal helpers
- **Map I/O Contract:** Always link `[[consumes::Artifact]]` so agents understand data shapes
- **Include Usage Examples:** Show how to actually use the code

## Semantic Links

| Link Type | Target | Description |
|-----------|--------|-------------|
| `implements::Principle` | Principle | Theoretical principle this implements |
| `requires_env::Environment` | Environment | Required environment setup |
| `consumes::Artifact` | Artifact | Input data consumed |
| `produces::Artifact` | Artifact | Output data produced |
| `used_in::Workflow` | Workflow | Workflows that use this implementation |

## Critical Rule

**Never describe an input as just "a dictionary."** You must define an Artifact and link to it with proper schema documentation.

