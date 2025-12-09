# Node Type: Environment

## Definition

**Role:** Context (Prerequisite)

An **Environment** defines the hardware, OS, and dependencies required to run the repository. It captures all prerequisites needed before any workflow can execute.

## Purpose

- Documents system requirements (OS, GPU, RAM)
- Lists all dependencies with versions
- Provides installation instructions
- Prevents "it doesn't work on my machine" issues

## Key Characteristics

| Aspect | Description |
|--------|-------------|
| Type | Context |
| Function | Prerequisite |
| Scope | Hardware, OS, Dependencies |

## Required Metadata

| Field | Description |
|-------|-------------|
| Repo URL | GitHub repository URL from `metadata.json` |
| Domain(s) | Up to 3 domain tags, comma-separated |
| Last Updated | Datetime in `YYYY-MM-DD HH:MM GMT` format |

## Template Structure

```mediawiki
= Environment: {Name} =
[[Category:Environments]]

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
{Brief description of the environment setup and its purpose.}

== System Requirements ==
{| class="wikitable"
! Component !! Requirement !! Notes
|-
|| OS || {e.g., Linux, Ubuntu 20.04+} || {Additional notes}
|-
|| GPU || {e.g., NVIDIA with CUDA 11.8+} || {Memory requirements}
|-
|| RAM || {e.g., 16GB minimum} || {Recommended amount}
|}

== Dependencies ==
=== Core Dependencies ===
* {package_name} >= {version} - {Purpose}

=== Optional Dependencies ===
* {package_name} - {Purpose and when needed}

== Installation ==
<syntaxhighlight lang="bash">
# Installation commands
pip install {package}
</syntaxhighlight>

== Code References ==
'''GitHub Repository:''' [{repoUrl} {repo_name}]

'''Setup Files:'''
* [{repoUrl}/blob/{branch}/requirements.txt requirements.txt] - Python dependencies

== Related Pages ==
* [[required_by::Implementation:{Class_Name}]] - Implementations requiring this
* [[required_by::Workflow:{Workflow_Name}]] - Workflows requiring this
```

## Coverage Target

**Minimum:** 2-3 pages per repository

## Sources to Scan

- `README.md`
- `requirements.txt`
- `setup.py` / `pyproject.toml`
- `Dockerfile`
- `environment.yml`

## Semantic Links

| Link Type | Target | Description |
|-----------|--------|-------------|
| `required_by::Implementation` | Implementation | Code that requires this environment |
| `required_by::Workflow` | Workflow | Workflows that require this environment |

