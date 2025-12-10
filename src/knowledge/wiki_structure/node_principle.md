# Node Type: Principle

## Definition

**Role:** Theory (Library-Agnostic)

A **Principle** is a single, atomic theoretical principle or algorithm. It answers "What is this technique?" and "Why does it work?" It represents the theoretical foundation that is independent of any specific implementation.

## Purpose

- Explains the "Why" behind techniques
- Provides library-agnostic understanding
- Links theory to concrete implementations

## Key Characteristics

| Aspect | Description |
|:---|:---|
| Type | Theory |
| Function | Atomic, abstract explanation |
| Scope | Library-agnostic |
| Nature | Static, Theoretical, Mechanism |

## Template Structure

```mediawiki
= Principle: {Name} =
[[Category:Principles]]

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

== Definition ==
{Name} is a {technique/algorithm/pattern} used to {function}.

== Theoretical Basis ==
The internal logic operates as follows:
<math>{Formula}</math>

== Related Implementation ==
* [[implemented_by::Implementation:{Class_A}]] - Primary implementation
* [[implemented_by::Implementation:{Class_B}]] - Alternative implementation

== Related Pages ==
* [[uses_heuristic::Heuristic:{Relevant_Wisdom}]] - Heuristics optimizing this theory
```

## Semantic Links

| Link Type | Target | Description |
|:---|:---|:---|
| `implemented_by::Implementation` | Implementation | Code that implements this principle (**Mandatory**) |
| `uses_heuristic::Heuristic` | Heuristic | Theoretical wisdom/optimization tips |

## Critical Constraints

1. **Executability:** Every Principle MUST link to at least one Implementation (`implemented_by`).
2. **Abstract Principles:** If a Principle is abstract (e.g., "Optimization"), link to the Base Class or Interface in the code.
