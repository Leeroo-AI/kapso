# P1 — Trajectory store + corpus import

**Goal:** the store exists per MD§3.4, and the ~64 archived campaigns are in
it. **Design sources:** MD§3.4 (structure, save, load), MD§3.4.1 (bundle
notes), MD§8 phase 1. **Depends on:** nothing. **Doubts:** D1, D2, D5, D7.

## Deliverables

`src/kapso/learning/` package skeleton; `trajectory_store.py`; the
`learning:` config block opened in `src/kapso/config.yaml` (only the keys
this phase reads — `trajectory_store: {remote, local}`; the block grows with
later phases, single-sourced per Rule 1); `kapso learn` CLI group with
`import` command; import of the corpus.

## Work items

1. **Package + config + CLI skeleton.** Create `src/kapso/learning/` (empty
   modules land per phase); add `learning.trajectory_store: {remote, local}`
   to `config.yaml` (MD§3.4: "no remote configured → the store is the local
   directory and everything still resolves"); wire a `kapso learn` subcommand
   group into the existing CLI entry (verify the actual entry —
   `src/kapso/cli.py` / `kapso.py` — at implementation; the CLI launches
   learning features as subprocesses so core stays light, Rule 4).
2. **Store write path** (MD§3.4): trajectory id `<task>/<stamp>_<lane>`;
   unpacked prefixes, one object per file, **manifest written last as the
   commit marker**; `trajectory.yaml`; `save_trajectory(result_or_path, …)` =
   gather → validate → hash → register (idempotent) → upload; the relbench
   adapter supplies gather paths (a mapping module under
   `benchmarks/relbench/`, not framework core).
3. **Store read path** (MD§3.4): exactly three functions, "no other door" —
   `manifest`, `resolve`, `open_ref`; refs resolve only through them.
4. **Corpus import**: enumerate the `.tgz` refs from
   `benchmarks/relbench/RESULTS.md` + a bucket listing (D1 scope); download,
   unpack, `save_trajectory` each (idempotency makes re-runs safe); write an
   import report (imported / failed-validation with named findings — Rule 2:
   a corrupt bundle raises per-item and is reported, not skipped silently).
   Store home per D2.
5. **(G) GATED — runner capture** (approval D5a): archive the ideation
   candidate pool + selector reasoning (today they die in temp worktrees,
   MD§3.4.1), workspace `.kapso`, and shared cache into future bundles.
   Touches `benchmarks/relbench/runner.py` / evolve archive path — diff shown
   before merge. Not needed for the historical corpus (mining's gap policies
   cover its absence).

## Tests (Rule 9 — the regression each catches)

- Double-save idempotency (re-import must not duplicate or clobber).
- Manifest-last semantics: a bundle without its manifest is invisible to
  `resolve` (partial-upload safety).
- Corrupt `trajectory.yaml`/manifest **raises**; missing optional artifact
  returns the documented default (MD§3.4 contract).
- `open_ref` refuses paths outside the store (adoption-before-mining: cards
  only ever cite store-resolvable ids, MD§4.1).

## Done gate

Corpus imported per D1; import report reviewed; tests green and added to the
curated suite (D7).
