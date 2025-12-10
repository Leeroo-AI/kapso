# Node Type: Heuristic

## Definition

**Role:** Wisdom (Tactical Intuition)

A **Heuristic** captures tribal knowledge, decision frameworks, and optimizations. It represents the "Art" of engineering - practical wisdom not explicitly stated in documentation.

## Purpose

- Documents best practices and gotchas
- Captures optimization techniques
- Provides decision frameworks (X vs Y)

## Key Characteristics

| Aspect | Description |
|:---|:---|
| Type | Wisdom |
| Function | Tactical Intuition |
| Nature | The "Art" of Engineering |
| Graph Role | **Leaf Node** (Target of `uses_heuristic`) |

## Template Structure

```mediawiki
= Heuristic: {Name_of_Insight} =
[[Category:Heuristics]]

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

== The Insight (Rule of Thumb) ==
{The core advice.}

== Reasoning ==
{Why this works.}

== Usage ==
This heuristic is used by:
* {List of Workflows/Principles/Implementations that link here}
```

## Semantic Links

*None (Leaf Node)*.
This node is the **target** of `uses_heuristic` links from other nodes.
