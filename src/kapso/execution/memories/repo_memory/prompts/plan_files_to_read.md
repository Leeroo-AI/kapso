You are inferring a repository's architecture and algorithms.

You will choose up to {{max_files_to_read}} files to read. Choose files that maximize:
- understanding of core algorithms/ideas
- entrypoints (how to run)
- configuration and evaluation contracts

Selection rules (high leverage for memory quality):
- Always try to include ALL repo key files and at least one entrypoint.
  - If there are many entrypoints, include the 1-3 most central ones (e.g., `main.py`, `cli.py`, `app.py`).
- Include the core "business logic" modules imported or called by the entrypoint(s).
- Include evaluation / scoring / grading contracts if present (e.g., files mentioning "SCORE", "eval", "grader", "metrics").
- Include tests if present (e.g., `tests/`, `test_*.py`) because they often encode invariants and edge cases.
- Prefer small, human-authored source files and docs over generated data or vendored code.
- Avoid: lockfiles, large datasets, compiled artifacts, logs, vendored deps.

You MUST return valid JSON only:
{
  "files_to_read": [
    {"path": "README.md", "why": "..."}
  ]
}

Repo key files: {{key_files_json}}
Repo entrypoints: {{entrypoints_json}}
Sample file list (paths): {{files_json}}

