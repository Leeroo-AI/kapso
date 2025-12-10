# Node Type: Workflow

## Definition

**Role:** Recipe (High-value "Job to be Done")

A **Workflow** is an ordered sequence of Principles that delivers a high-value business outcome. It is temporal (Start → End) and represents what users actually want to accomplish.

## Purpose

- Documents end-to-end processes
- Maps the "jobs to be done" in the repository
- Shows how Principles combine to solve real problems
- Provides execution diagrams for visual understanding

## Key Characteristics

| Aspect | Description |
|:---|:---|
| Type | Recipe |
| Function | High-value "Job to be Done" |
| Scope | Ordered sequence of Principles |
| Nature | Temporal (Start → End) |

## Template Structure

```mediawiki
= Workflow: {Name} =
[[Category:Workflows]]

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Identifier
|| {unique_identifier}
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
{Description of the end-to-end process derived from Intent.}

== Execution Steps ==
=== Step 1: {Step_Name} ===
[[step::Principle:{Step_1_Abstract}]]

{Detailed description of what happens in this step.}

=== Step 2: {Step_Name} ===
[[step::Principle:{Step_2_Abstract}]]

== Related Pages ==
* [[uses_heuristic::Heuristic:{Relevant_Wisdom}]] - Heuristics guiding this process
```

## Semantic Links

| Link Type | Target | Description |
|:---|:---|:---|
| `step::Principle` | Principle | A step in the workflow |
| `uses_heuristic::Heuristic` | Heuristic | Process wisdom/guidance |
