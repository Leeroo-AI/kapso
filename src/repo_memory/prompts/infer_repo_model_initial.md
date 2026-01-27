You are inferring repository memory for an automated coding system.

Return ONLY valid JSON in the **Book** format below. Every claim MUST include evidence.

Evidence format: Use LINE NUMBERS to reference code.
- Each evidence item needs: `path` (file path) and `line` (line number)
- Optionally include `description` to briefly describe what the line shows
- Line numbers are shown as "N| content" in the file contents below
- Use the exact file path shown in the FILE header (e.g., `=== FILE: main.py ===` -> path is `main.py`)

Example evidence:
```json
"evidence": [
  {"path": "utils.py", "line": 65, "description": "Model size validation"},
  {"path": "utils.py", "line": 72, "description": "Downloads files if missing"}
]
```

Required JSON schema (RepoMemory V2 format):
{
  "summary": "High-level what the repo does (1-2 sentences)",
  "sections": {
    "core.architecture": {
      "title": "Architecture",
      "one_liner": "System design and module structure",
      "claims": [
        {
          "kind": "algorithm|architecture|contract|deployment|other",
          "statement": "...",
          "confidence": 0.0,
          "evidence": [{"path": "path/in/repo.py", "line": 42, "description": "brief description"}]
        }
      ]
    },
    "core.entrypoints": {
      "title": "Entrypoints",
      "one_liner": "How to run the application",
      "content": [{"path": "main.py", "how_to_run": "python main.py --help"}]
    },
    "core.where_to_edit": {
      "title": "Where to edit",
      "one_liner": "Key files for modifications",
      "content": [{"path": "src/foo.py", "role": "core algorithm implementation"}]
    },
    "core.invariants": {
      "title": "Invariants",
      "one_liner": "Contracts, constraints, and assumptions",
      "claims": []
    },
    "core.testing": {
      "title": "Testing",
      "one_liner": "How to run tests and validate changes",
      "claims": []
    },
    "core.gotchas": {
      "title": "Gotchas",
      "one_liner": "Common pitfalls and sharp edges",
      "claims": []
    },
    "core.dependencies": {
      "title": "Dependencies",
      "one_liner": "Key dependencies and environment notes",
      "claims": []
    },
    "opt.<slug>": {
      "title": "Optional Section Title",
      "one_liner": "One line summary for TOC",
      "claims": []
    }
  }
}

Rules:
- Include at least the core.* section keys shown above (they may be empty).
- Optional sections MUST use IDs starting with `opt.` (e.g., `opt.payment_flow`).
- Only include `claims` in sections that need evidence-backed statements.

Quality rubric (this is what makes the memory useful):
- Your goal is NOT to list files. Your goal is to capture the repo's *meaning* so a coding agent can work quickly.
- Prefer claims that answer:
  - What does the repo do end-to-end? (data in -> transforms -> output)
  - What are the key contracts? (inputs, outputs, required keys/fields, formats)
  - What are the sharp edges? (silent fallbacks, default values, "0.0" coercions, empty behavior)
  - How do we quickly validate changes? (how to run, what output indicates success/score)
- Populate these sections when evidence exists in FILE CONTENTS:
  - core.architecture: 3-6 claims describing module responsibilities + data flow.
  - core.invariants: 2-5 claims describing input/output contracts and evaluation signals.
  - core.gotchas: 1-4 claims describing surprising behavior and edge cases.
  - core.testing: 1-3 claims describing how to run a sanity check or tests.
  - core.dependencies: 1-3 claims describing notable deps / env vars / services.
- Keep it concise: prefer ~10-25 total claims. Avoid generic trivia.

Section definitions (avoid misplacing claims):
- core.architecture: module responsibilities + data flow (NOT dependencies).
- core.entrypoints: how to run the app/CLI (commands, scripts).
- core.where_to_edit: key files to modify + their roles.
- core.invariants: stable contracts (types, required keys, output formats like "SCORE:").
- core.testing: how to validate changes (tests or a quick manual run).
- core.gotchas: surprising behavior / edge cases (defaults, fallbacks, coercions). NOT dependencies.
- core.dependencies: libraries/env vars/services (e.g., evidence from requirements.txt, pyproject.toml, package.json).

RepoMap key files: {{repo_map_key_files_json}}
RepoMap entrypoints: {{repo_map_entrypoints_json}}

FILE CONTENTS (with line numbers - reference by line number in evidence):
{{files_payload}}
