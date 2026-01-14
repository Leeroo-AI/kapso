You are updating repository memory after code changes.

Return ONLY valid JSON in the RepoMemory V2 format and keep it evidence-backed:
- Preserve previous sections/claims if still supported.
- Update/add/remove claims and optional `opt.*` sections as needed based on the diff.
- Every NEW or MODIFIED claim MUST include evidence quotes from the provided CHANGED FILE CONTENTS.
- Unchanged claims may keep their existing evidence (even if their source file isn't in the changed set).

Evidence rules (to avoid validation failures):
- For each evidence item: `path` MUST be the file that contains the `quote`.
- Use the exact file path shown in the CHANGED FILE header (e.g., `=== FILE: src/foo.py ===`).
- NEVER include the `=== FILE: ... ===` header in the quote.
- Prefer single-line, short quotes (no newlines; ideally <120 chars).
- IMPORTANT: Do NOT "rewrite" a quote to make it single-line.
  - Instead, choose a shorter substring that already exists in the file.
- IMPORTANT: Do NOT invent pseudo-code as evidence.
  - Bad (usually not present verbatim): `if not data: return []`
  - Good: quote an actual line like `if not os.path.exists(filepath):` or `return []` that exists exactly.
- If you cannot find an exact quote that supports a NEW/MODIFIED claim, REMOVE the claim.
- For multi-line code (common in configs / function calls), do NOT collapse it into a fake one-liner quote.
  - Instead, include MULTIPLE evidence items, each quoting a real single line.

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
- Every NEW or MODIFIED claim must be grounded in an EXACT quote from CHANGED FILE CONTENTS.

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

PREVIOUS MODEL (may contain stale items):
{{previous_model_json}}

CHANGED FILE CONTENTS (authoritative for NEW/MODIFIED claims' evidence quotes):
{{changed_payload}}

