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

== Related Pages ==
=== Required By ===
* [[required_by::Implementation:{Implementation_Name}]]

```

## Semantic Links

*None (Leaf Node)*.
This node is the **target** of `requires_env` links from Implementations.
