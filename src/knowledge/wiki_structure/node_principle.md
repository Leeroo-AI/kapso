# Node Type: Principle

## Definition

**Role:** Theory (Library-Agnostic)

A **Principle** is a single, atomic theoretical principle or algorithm. It answers "What is this technique?" and "Why does it work?" It represents the theoretical foundation that is independent of any specific implementation.

## Purpose

- Explains the "Why" behind techniques
- Provides library-agnostic understanding
- Serves as reusable theoretical knowledge
- Links theory to concrete implementations

## Key Characteristics

| Aspect | Description |
|--------|-------------|
| Type | Theory |
| Function | Atomic, abstract explanation |
| Scope | Library-agnostic |
| Nature | Static, Theoretical, Mechanism |

## The Textbook Test

Ask: "Could I write a Wikipedia article about this specific logic without mentioning this repository?"

- **YES** → It's a Principle ✓
- **NO** → It's likely a Workflow

## Critical Constraints

1. **NO Business Logic:** Don't write "We use this to track customer churn" (Workflow). Write "This calculates the probability of attrition" (Principle).
2. **NO Code:** Use `<math>`, logic tables, or pseudo-code only.
3. **Atomic Scope:** If it requires 3 unrelated Principles to function, it's likely a Workflow.

## Two Types of Principle Pages

### A. Workflow Step Principle
- Used as a step in a Workflow via `[[step::Principle:repo_name/X]]`
- **MUST** have a "Related Implementation" section
- **MUST** link to at least one Implementation page
- Required because workflows must be executable

### B. Standalone Principle
- Theoretical principle not directly used as a workflow step
- "Related Implementation" section is optional
- May exist purely for theoretical context

## Required Metadata

| Field | Description |
|-------|-------------|
| Repo URL | GitHub repository URL from `metadata.json` |
| Domain(s) | Up to 3 domain tags, comma-separated |
| Last Updated | Datetime in `YYYY-MM-DD HH:MM GMT` format |

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
{Name} is a {technique/algorithm/pattern} used to {function}. It is distinct from {Alternative} because {Reason}.

== Theoretical Basis (The Mechanism) ==
The internal logic operates as follows:
<math>
{Formula}
</math>
* **Principle:** {e.g., "Minimizes the loss function via..."}
* **Key Constraint:** {e.g., "Requires differentiable functions."}

== Related Implementation ==
{REQUIRED if this Principle is used as a step in a Workflow.}
* [[realized_by::Implementation:{repo_name}/{Class_A}]] - Primary implementation
* [[realized_by::Implementation:{repo_name}/{Class_B}]] - Alternative implementation

== Related Pages ==
* [[related_to::Principle:{repo_name}/{Parent_Theory}]] - Parent or related theory
* [[step_of::Workflow:{repo_name}/{Workflow_Name}]] - Workflow using this principle
```

## Coverage Target

**Minimum:** 8-12 pages per repository (including ALL workflow steps)

## Semantic Links

| Link Type | Target | Description |
|-----------|--------|-------------|
| `realized_by::Implementation` | Implementation | Code that implements this principle |
| `related_to::Principle` | Principle | Related theoretical concepts |
| `step_of::Workflow` | Workflow | Workflows that use this principle |

## Workflow Executability Rule

**Every Principle that serves as a workflow step MUST have at least one corresponding Implementation page linked via `[[realized_by::Implementation:repo_name/Y]]`.**

This ensures workflows are executable:
- The Principle explains the theory (WHY)
- The Implementation provides the actual code (HOW)

