# Kaggle competition benchmark

Runs kapso as an autonomous contestant on a Kaggle competition, from a URL to a
public leaderboard score.

```bash
# 1. URL -> run root (one codex call: downloads data, authors the statement)
python -m benchmarks.ioai2026.preflight \
    --url https://www.kaggle.com/competitions/<slug>/overview --root ~/run_root

# 2. run root -> scored campaign (k and hours come from config run_defaults)
python -m benchmarks.ioai2026.runner --root ~/run_root
```

The run root is the contract between the two stages:

```
<root>/task/
├── dataset/
│   ├── statement.md     <- authored by the preflight; the ONLY per-competition input
│   └── <competition data>
└── kaggle.json          <- {"competition": "<slug>"}
```

At launch the runner also stages `RULES.md` (organizers' binding rules),
`KAGGLE_CLI.md` (the repo's kaggle-cli-submission skill — the submit playbook
every coding-agent CLI reads by path, since session clones can't load skills
natively) and `kernel_slots.py` (per-pool priority queues over Kaggle's session
limits — pushes and scoring re-runs both consume sessions) into `task/`.

`config.yaml` is fixed per campaign, so the preflight's `statement.md` is the
entire per-competition surface. Box provisioning and setup live in `infra/`;
findings from live runs are in `RUN_FINDINGS.md`; the proposed per-round K
schedule is in `K_RAMP_SPEC.md`.

---

## ⚠️ Merge safety — read before merging this branch

Most of the work here is **Kaggle-only** and merges freely. A few pieces are
not. This section exists so a future merge does not silently change another
benchmark or quietly undo a deliberate choice.

**Scope of this section:** the Kaggle-automation work (2026-07-28 → 07-30,
`2786cd40` … `af5a1160`). The branch *also* carries earlier core features from
the IOAI campaign — codex coding-agent adapter, K-way node expansion, the
ideation lens planner, shared cache / artifact registry, insured finalization —
which are separate concerns with their own blast radius and are not covered here.

### 1. Two commits touch framework core (`src/kapso/`)

| Commit | Change | Blast radius |
|---|---|---|
| `9243724b` | Deleted `ENSEMBLE_MEMBER_TIME_FRACTION` / `ENSEMBLE_SELECTOR_TIME_FRACTION` / `ENSEMBLE_SELECTOR_MIN_SECONDS`; members and the selector each get the full ideation clamp, and the selector recomputes it after the members finish | **Every ensemble benchmark** (ioai, ioai_tasks, posttrain, relbench). Each role now gets the whole ideation ceiling instead of 70% / 30% of it, so ideation can take up to ~2× its ceiling end to end. Intentional (Rule 7) — the split was starving members and killing the selector — but re-check those benchmarks' `ideation_timeout` if their economics matter. |
| `63eda5d7` | `ideation_candidates_per_member` read from `search_strategy.params` instead of a module literal | Safe. The module constant remains the sourced default (`2` — the old literal), so behaviour is unchanged unless a config sets it. |

Every other change in the Kaggle-automation work is confined to
`benchmarks/ioai2026/` and `tests/test_kaggle_benchmark.py`.

### 2. `session_budget` has a different shape here than everywhere else

`af5a1160` sized the finalization reserve to one submission round trip, because
a Kaggle code competition's ship action is a 10–20 min cloud round trip, not a
write-up. That made the fraction/min/max triple meaningless here:

| Benchmark | `session_budget` keys |
|---|---|
| **kaggle** | `finalization_reserve_minutes`, `insured_reserve_minutes`, `guard_minutes` |
| ioai, ioai_tasks, posttrain | `ideation_fraction`, `ideation_min_seconds`, `implementation_fraction`, `implementation_min_seconds`, `finalization_reserve_fraction`, `finalization_reserve_min_minutes`, `finalization_reserve_max_minutes`, `guard_minutes` |

**Do not mechanically unify these.** The kaggle shape encodes "hold enough clock
to ship once"; the others encode per-phase fractions. Converging them means
deciding which model is right for each benchmark, not a find-and-replace.

### 3. `shape_session_timeouts` is a per-benchmark copy, and kaggle's now differs

Four runners define a function of this name (`kaggle`, `ioai`, `ioai_tasks`,
`posttrain`). Kaggle's no longer applies per-phase fractions — it bounds the
configured ceilings by the run and lets the strategy's dynamic clamp be the only
enforcer. **Copying one runner's version over another's will break it**, because
kaggle's reads no fraction keys and the others require them.

### 4. Deliberate choices that look like bugs

Reverting any of these silently regresses behaviour:

- **`runner.py --coding-agent` defaults to `None`** (`3ddb282f`). The orchestrator
  treats an explicit value as an override, so restoring the old `"claude_code"`
  default would silently replace the codex-primary `coding_agent` block with a
  bare default-model claude agent.
- **`KaggleNotebookHandler.__init__` requires `insured_reserve_seconds`**
  (`af5a1160`). Any other construction site or fake must pass it.
- **Lanes learn from each other through Kaggle, not through a shared file** —
  the handler points them at `kaggle competitions submissions` / `kernels pull`,
  which is the one view every lane sees identically (each lane works in its own
  git clone, so a sibling's path does not resolve). `best_score.log` survives
  for a different job: it records the PUBLIC scores actually banked, and it is
  what `deliverable_ready_reserve_seconds()` reads.
- **`benchmarks/ioai2026/data/` was deleted** (`2786cd40`) — `prepare_task1.py`,
  `prepare_task2.py` and the hand-written statements are superseded by the
  preflight. A merge that resurrects them reintroduces per-task Python that the
  URL-driven flow no longer has any caller for.
- **Session ceilings are 14400s and `run_defaults` are k=8 / 2h** — deliberately
  larger than any run we launch, so the run budget is the only limit.

### 5. Harvest depends on Kaggle-specific facts

`af5a1160`'s harvest encodes things that are true of the Kaggle CLI and would be
wrong to generalise:

- `-v` is **mandatory** for code-competition submissions, and the CLI exposes a
  kernel's version **nowhere** (`kernels list --format json` and `kernels status`
  both omit it) — hence the downward version probe. Upward probing ships the
  *oldest* version.
- Kernel discovery is the **union** of local `kernel-metadata.json` files and
  `kernels list -m` filtered to the run window. Local alone is insufficient: a
  lane can push a kernel and record nothing (run 3's lane 3 did).
- Programmatic code-competition submission works here because auth is a
  **`KGAT_` token in `~/.kaggle/access_token` with kaggle CLI ≥ 2.2**. The legacy
  `~/.kaggle/kaggle.json` username+key path returns `403 Forbidden` on
  `CreateCodeSubmission`. Do not "simplify" auth back to `kaggle.json` — it turns
  the autonomous loop into a manual one.
