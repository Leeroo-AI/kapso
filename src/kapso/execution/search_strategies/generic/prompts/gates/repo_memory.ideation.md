### RepoMemory Access (MCP Tools)
The repository has a semantic memory that captures architecture, gotchas, and key patterns.

- **get_repo_memory_summary**: Get the summary and table of contents
  - Use this first to understand what sections are available
  - Example: `get_repo_memory_summary()`

- **get_repo_memory_section**: Get detailed content for a specific section
  - Use this to dive deep into architecture, gotchas, etc.
  - Example: `get_repo_memory_section(section_id="core.architecture")`
  - Available sections: core.architecture, core.entrypoints, core.where_to_edit, core.invariants, core.testing, core.gotchas, core.dependencies

- **list_repo_memory_sections**: List all available section IDs
  - Example: `list_repo_memory_sections()`
