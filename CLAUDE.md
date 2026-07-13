# CLAUDE.md

Engineering rules for this repository. These override any default behavior — follow them exactly as written.

## 1. Config-like constants live in the config, single-sourced — never scattered literals

A tunable, operational value — a threshold, budget, size/limit, count, timeout/interval, ratio, schedule, model name or LLM param, port/host/url, retry count, cache/state path — belongs in the config (`src/kapso/config.yaml`, loaded via `load_config()` / `load_mode_config()` in `src/kapso/core/config.py`), read at the call site or threaded to it. Do NOT bury such a value as a literal in a module, and NEVER re-hardcode a default that duplicates an existing config value: a copied literal silently drifts from its config home — the exact bug class this rule exists to kill.

When a value genuinely needs a module-level or dataclass default (for direct construction / tests), SOURCE it from the canonical config — not a fresh literal — so there is ONE source of truth and a default change can never leave two copies out of sync; the live per-run override still threads from `load_config()`.

This does NOT cover STRUCTURAL constants — regex / format / prompt strings, JSON or enum keys, sentinels, array indices, version strings, HTTP status codes, math constants — which are not user knobs and stay in code. Companion to Rule 3: Rule 3 says config comes from the file (never an env var); this rule says config-like values must actually BE in the config, not duplicated as literals next to it.

## 2. No try/except, no fallbacks — validate explicitly, fail loud

No try/except anywhere in the code, and no fallback-style error swallowing. Validate inputs with explicit checks (`isinstance`, `.exists()`, presence checks, regex) and let genuine errors propagate. A missing file may return a documented default; a corrupt one must raise. This is strict even for parsing on-disk JSON/JSONL and model JSON output: a malformed line raises, it is not skipped.

## 3. No configuration via environment variables

All configuration is read from the config file (`config.yaml`, passed explicitly via `--config` / `config_path`), never from environment variables. Never read `os.environ` / `getenv`. Secrets are not read by our code — provider SDKs read their own credentials; the config file holds no secrets.

## 4. All imports at the top of the file

Every `import` / `from … import` goes at the top of its module — no function-local or lazy imports. To keep the package installable without heavy optional deps, isolate those deps in modules that are only loaded when that feature runs, and have the core CLI launch them as subprocesses rather than importing them.

## 5. Naming — self-contained, never index-style

Code identifiers must be descriptive of what they do. NEVER use index-style names (M1/M2/…, C18, etc.) in code (functions, classes, modules, vars). Docstrings/comments may cite a design doc's index names for provenance ("M4 §4.1"), but code's own names stay descriptive.

## 6. Never truncate a message bound for an LLM call — above all a user message

Any content that goes INTO a model prompt must be passed in FULL. NEVER clip it with a character cap (`text[:N]`), a "first non-blank line", a "head" slice, or any other length limit on the way to an LLM call. This is strictest for the user's own messages (prompts, pasted code/errors): the model must see the whole thing — intent and detail do not always live in the first N characters. This covers every prompt-building path (search strategies, researcher, knowledge-base prompts). A model-produced ABSTRACTION (e.g. a judge's one-line label) is fine — that is the model's own summary, not us cutting the input. If a prompt is genuinely too large for the context window, WINDOW it (segment into multiple complete calls) or bubble the problem up — never silently truncate. Length caps for non-LLM purposes — display (dashboards, ledgers), git short-shas, id/slug minting, prefix classifiers that only inspect a head — are allowed; the rule is specifically about data crossing into a model call.

## 7. No backward compatibility during development — implement the new design cleanly

This codebase is pre-release and still in active design flux. When a design decision changes, implement the NEW design directly — do NOT keep the old behavior alive for compatibility. No migration shims, deprecated aliases, dual code paths, "support both the old and new format" branches, or versioned fallbacks: delete the superseded code/format and move its callers and tests to the new shape. Prior on-disk state written under an earlier design may be re-derived or discarded — there are no external consumers to protect yet. Backward compatibility becomes a concern only once we ship; until then the codebase tracks the current design with no legacy weight.

## 8. Commit after each feature change

When a feature change is complete and working, commit it before moving on — don't let finished work pile up uncommitted. Each commit is one coherent, self-contained change (the feature/fix plus its tests and docs), with the suite green at that point; keep them atomic — never batch several unrelated changes into one commit. Use a clear message describing the change and end it with the Co-Authored-By trailer. Commit-as-you-go is the standing default — commit after each change, documentation and design-doc edits included, without pausing to ask each time. Pushing stays explicit on feature/worktree branches (push those only when the user asks); the default branch `main` is the exception — its commits are pushed as you go (Rule 11).

## 9. Every test must earn its place

A test exists to protect real behavior or to pin down a tricky contract — never to raise a coverage number. We run free by mocking the provider boundary, so a test that only exercises mocks proves nothing; assert on the actual shape a module produces, the fail-loud paths (a corrupt line raises, a missing file returns the documented default), and the contracts the design pins down. If you can't say what regression a test would catch, don't write it.

## 10. Minimalism — every line fights for its existence

Keep the code and the on-disk JSON minimal: each line, field, and file must earn its place. When you see an opportunity for dramatic simplification — a move that deletes a layer, unifies two shapes, collapses a special case, or makes the design inevitable — bring it up rather than reaching for an incremental patch that adds code. Given Rule 7 (no legacy weight) we are free to take the simplifying move directly; prefer deleting code over guarding it.

## 11. Commits on main are pushed, not left local

A commit made directly on the default branch (`main`) is pushed in the same step — never leave `main` ahead of `origin/main` with local-only commits. This is the standing exception to Rule 8's "push only when the user asks": that default still governs feature and worktree branches, but anything that lands on `main` ships to the remote immediately (still atomic and suite-green per Rule 8 — push right after the commit). If the push is rejected because someone else advanced `origin/main`, pull/rebase and retry; never force-push `main`.
