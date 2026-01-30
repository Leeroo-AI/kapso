You are updating repository memory after code changes.

Return ONLY valid JSON in the RepoMemory V2 format with line-number-based evidence:
- Preserve previous sections/claims if still supported.
- Update/add/remove claims and optional `opt.*` sections as needed based on the diff.
- Every NEW or MODIFIED claim MUST include evidence with line numbers from the provided CHANGED FILE CONTENTS.

CRITICAL - Handling changed files:
- If a file appears in CHANGED FILE CONTENTS, you MUST update ALL evidence line numbers for that file.
- The old line numbers in PREVIOUS MODEL are STALE for changed files - do NOT copy them.
- Look up the correct line number from CHANGED FILE CONTENTS (format: "N| content").
- If the code referenced by a claim was deleted or no longer exists, REMOVE the claim entirely.
- If the code moved to a different line, update the line number to the new location.

Evidence format: Use LINE NUMBERS to reference code.
- Each evidence item needs: `path` (file path) and `line` (line number)
- Optionally include `description` to briefly describe what the line shows
- Line numbers are shown as "N| content" in the file contents below
- Use the exact file path shown in the CHANGED FILE header (e.g., `=== FILE: src/foo.py ===`)

Example evidence:
```json
"evidence": [
  {"path": "utils.py", "line": 65, "description": "Model size validation"},
  {"path": "utils.py", "line": 72, "description": "Downloads files if missing"}
]
```

Schema (RepoMemory V2):
{
  "summary": "...",
  "sections": {
    "core.architecture": {"title": "...", "one_liner": "...", "claims": [...]},
    "core.entrypoints": {"title": "...", "one_liner": "...", "content": [...]},
    "core.where_to_edit": {"title": "...", "one_liner": "...", "content": [...]},
    "core.invariants": {"title": "...", "one_liner": "...", "claims": [...]},
    "core.testing": {"title": "...", "one_liner": "...", "claims": [...]},
    "core.gotchas": {"title": "...", "one_liner": "...", "claims": [...]},
    "core.dependencies": {"title": "...", "one_liner": "...", "claims": [...]},
    "opt.<slug>": {"title": "...", "one_liner": "...", "claims": [...]}
  }
}

Quality rubric (keep the memory actionable for coding agents):
- Update the *meaning* of the repo, not just the file list.
- If the diff changes behavior, update the relevant semantic sections:
  - core.architecture: data flow or module responsibility changes
  - core.invariants: input/output keys, formats, evaluation strings (e.g., "SCORE:")
  - core.gotchas: new fallbacks, defaults, coercions, edge cases
  - core.testing: new validation steps or tests
  - core.dependencies: new deps / env vars / services
- Keep it concise: avoid adding many low-value claims.
- Every NEW or MODIFIED claim must reference line numbers from CHANGED FILE CONTENTS.

Section definitions (avoid misplacing claims):
- core.architecture: module responsibilities + data flow (NOT dependencies).
- core.entrypoints: how to run the app/CLI (commands, scripts).
- core.where_to_edit: key files to modify + their roles.
- core.invariants: stable contracts (types, required keys, output formats like "SCORE:").
- core.testing: how to validate changes (tests or a quick manual run).
- core.gotchas: surprising behavior / edge cases (defaults, fallbacks, coercions). NOT dependencies.
- core.dependencies: libraries/env vars/services (e.g., evidence from requirements.txt, pyproject.toml, package.json).

DIFF SUMMARY:
{{diff_summary}}

PREVIOUS MODEL (may contain stale line numbers for changed files - verify against CHANGED FILE CONTENTS):
{{previous_model_json}}

CHANGED FILE CONTENTS (with line numbers - use these as the source of truth for evidence):
{{changed_payload}}
