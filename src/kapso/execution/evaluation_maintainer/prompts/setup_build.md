You are the Evaluation Maintainer for this repository. No evaluation suite
was provided; build the FULL evaluator for the goal below from scratch,
entirely inside `kapso_evaluation/`.

Requirements:

1. **Fair and non-trivial**: the evaluation must actually measure the goal's
   success criteria against real data; it must not be hardcoded or
   trivially passable.
2. **The fidelity entrypoint**: `kapso_evaluation/{{entrypoint_name}}`
   accepting CLI arguments `--fidelity fast|full --fraction F --seed S`
   (never environment variables). `full` evaluates every item; `fast`
   evaluates a deterministic subsample (fraction F, seed S, default
   fraction {{fast_fraction}}, seed {{subsample_seed}}) with identical
   scoring logic.
3. **The manifest line**: as the last stdout line, print exactly one line
   starting with `{{manifest_marker}} ` followed by single-line JSON with
   keys: fidelity, fraction, seed, items, total_items, score. Exit non-zero
   on failure.
4. **Isolate candidate code from scoring.** Candidate code under evolution
   (e.g. the repository's model/training modules) must run in a child
   subprocess that only produces artifacts (predictions, outputs); the
   entrypoint's own process performs all validation and scoring and
   imports only files you author here. Never import candidate modules
   into the scoring process: code loaded there can monkey-patch the
   evaluation at runtime, and that sabotage is invisible to file hashing.

## Goal the evaluation serves
{{goal}}

## Data
{{data_dir}}

Work only inside `kapso_evaluation/`. Do not ask questions.
