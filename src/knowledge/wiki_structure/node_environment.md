# Node Type: Environment

## Definition

**Role:** Context (Prerequisite)

An **Environment** defines the hardware, OS, and dependencies required to run the repository.

## Purpose

- Documents system requirements
- Lists all dependencies with versions
- Prevents "it doesn't work on my machine" issues

## Key Characteristics

| Aspect | Description |
|:---|:---|
| Type | Context |
| Function | Prerequisite |
| Graph Role | **Leaf Node** (Target of `requires_env`) |

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

== System Requirements ==
{| class="wikitable"
! Component !! Requirement
|-
| OS || {e.g., Linux}
|-
| GPU || {e.g., CUDA 11.8}
|}

== Dependencies ==
* {package_name} >= {version}

== Usage ==
This environment is required by:
* {List of Implementations that link here}
```

## Semantic Links

*None (Leaf Node)*.
This node is the **target** of `requires_env` links from Implementations.
