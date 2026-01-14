You are inferring repository memory for an automated coding system.

Return ONLY valid JSON in the **Book** format below. Every claim MUST include evidence.

CRITICAL: Evidence quotes must be EXACT VERBATIM substrings that appear in the file.
- Copy quotes character-for-character from the file content below
- Do NOT paraphrase, summarize, or modify quotes in any way
- The quote must exist as a continuous substring in the file
- Shorter quotes (e.g., function signatures, class names) are safer than long ones

Evidence rules (to avoid validation failures):
- For each evidence item: `path` MUST be the file that contains the `quote`.
- Use the exact file path shown in the FILE header (e.g., `=== FILE: main.py ===` -> path is `main.py`).
- NEVER include the `=== FILE: ... ===` header in the quote.
- Prefer single-line, short quotes (no newlines; ideally <120 chars).
- IMPORTANT: Do NOT "rewrite" a quote to make it single-line.
  - Instead, choose a shorter substring that already exists in the file.
  - Example: prefer quoting `Accuracy as a float in [0, 1]` (verbatim) rather than inventing `Returns accuracy as a float in [0, 1]`.
- IMPORTANT: Do NOT invent pseudo-code as evidence.
  - Bad (usually not present verbatim): `if not data: return []`
  - Good: quote an actual line like `if not os.path.exists(filepath):` or `return []` that exists exactly.
- If you cannot find an exact quote that supports a claim, REMOVE the claim.
  It is OK for sections (including core.gotchas) to be empty.
- For multi-line code (common in configs / function calls), do NOT collapse it into a fake one-liner quote.
  - Instead, include MULTIPLE evidence items, each quoting a real single line.
  - Example for a LoRA config:
    - evidence: [{"path":"main.py","quote":"r=8,"},{"path":"main.py","quote":"lora_alpha=8,"}]

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
          "evidence": [{"path": "path/in/repo.py", "quote": "EXACT verbatim substring from file"}]
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

FILE CONTENTS (authoritative - copy quotes EXACTLY from here):
{{files_payload}}

