# Budgeted E2E track

The standing end-to-end vehicle for the budget-aware experimentation design
(`docs/evolve/budget-aware-experimentation.mdx`, "End-to-end verification
track"). Synthetic Spaceship-Titanic data (deterministic, no Kaggle
credentials), real agents, real budgets: a run spends coding-agent credits.

Every knob lives in `config.e2e.yaml` (three modes). The runner stages a
fresh campaign, spawns the run under a **hard wall of 2× the mode's time
budget**, then verifies the milestone invariants mechanically from the run
artifacts (checkpoint, evaluation registry, result). Exit 0 = all checks
pass, 1 = a check failed, 2 = hard wall breached.

Run from the repo root, in an environment with kapso's dependencies and
coding-agent credentials configured:

```bash
# M1/M5/M6 rows: full 45-min budgeted campaign, fidelity on
python examples/ml_model_development/e2e/run_e2e.py --mode E2E_BUDGET

# M2 row: simulated crash mid-campaign, then resume (clock continues)
python examples/ml_model_development/e2e/run_e2e.py --mode E2E_BUDGET --interrupt-after-seconds 300
python examples/ml_model_development/e2e/run_e2e.py --mode E2E_BUDGET --resume

# M3 row: budget top-up on resume (explicit argument beats the config block)
python examples/ml_model_development/e2e/run_e2e.py --mode E2E_BUDGET --resume --top-up-minutes 60

# M4 row: 8-min gate — reserve gate and deadline clamps must fire,
# campaign must still end with a checked-out artifact
python examples/ml_model_development/e2e/run_e2e.py --mode GATE_BUDGET

# M6 byte-equivalence row: fidelity off is a full passthrough
python examples/ml_model_development/e2e/run_e2e.py --mode E2E_BUDGET_FIDELITY_OFF

# Change-request row: a provided grader with a buried defect (rejects any
# submission mixing True/False — passes the single-class baseline, kills
# every honest model). The agent's only recourse is an
# <evaluation_change_request>; the leg verifies accept -> v2 -> the
# requester bridges first -> re-anchor.
python examples/ml_model_development/e2e/run_e2e.py --mode E2E_BUDGET --seed-eval-defect

# Re-run the checks on an existing run without spawning anything
python examples/ml_model_development/e2e/run_e2e.py --mode E2E_BUDGET --verify-only
```

What each mode must prove is encoded in `run_e2e.py`'s check list: durable
v2 checkpoint, per-node phase telemetry, budget-stop semantics
(`last_stop`/`stop_detail`), scores never blended across comparability
classes, the registry anchored on the evaluator head, the
`--fidelity/--fraction/--seed` contract honored, the committed best at
FULL evidence tier, and (gate mode) every agent call provably bounded
inside the campaign window.

Run artifacts land in `runs/<mode>/` (gitignored): the staged
`initial_repo/`, the campaign `workspace/`, and `result.json`.
