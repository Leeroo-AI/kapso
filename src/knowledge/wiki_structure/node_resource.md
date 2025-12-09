# Node Type: Resource

## Definition

**Role:** Container (Entry Point)

A **Resource** represents the repository, library, or paper being documented. It serves as the root entry point to the knowledge graph.

## Purpose

- Defines the scope of the documented system
- Provides high-level context about what the repository does
- Links to all other nodes in the knowledge graph

## Key Characteristics

| Aspect | Description |
|--------|-------------|
| Type | Container |
| Function | Entry Point |
| Scope | The entire repo, library, or paper |

## Typical Contents

- Repository name and URL
- High-level description of purpose
- Links to Environment nodes (prerequisites)
- Links to main Workflows (what users can do)
- Domain classification tags

## Notes

The Resource node is typically implicit in the structure - it's represented by the collection of all wiki pages for a given `repo_name`. The `repo_name` is derived from the GitHub URL in format `{owner}_{repo}`.

