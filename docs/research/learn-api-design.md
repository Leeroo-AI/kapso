# `Kapso.learn()` — the facade contract for learn-from-trajectories

**Status:** DESIGN for review. Packages trajectory learning as a
first-class `Kapso` capability, the peer of `evolve()` — one object, one
closed loop: `evolve → learn → evolve`. Written against the shipped
facade (`src/kapso/kapso.py` @ 09302f87: `evolve(...) -> SolutionResult`)
and the operating chain the CLI already exposes
(`kapso learn ingest` = import → mine → exam-before-lesson → lesson →
bank push).

---

## 1. The verb decision

`Kapso.learn(*sources)` today is the OLD knowledge-source pipeline
(`Source.Repo` → wiki pages → KG backends). The CLI already broke this
tie: `kapso learn` (CLI) is trajectory learning; the KG entry was
superseded (Rule 7, merge commit 20756e1d).

**Decision: the class follows the CLI.** `Kapso.learn()` becomes
trajectory learning — the agent learning from its own campaigns. The KG
pipeline keeps existing under an honest name: `Kapso.learn_knowledge()`
(same signature and behavior as today's `learn`, pure rename). No
overloading one verb across both: the two features have disjoint inputs,
options, results, and mental models — a shared name would be a pun, not
a unification. (`Source.Solution` in the KG path is unrelated to
campaign trajectories; the rename dissolves the collision.)

## 2. The contract

```python
def learn(
    self,
    source: Union[SolutionResult, str],
    *,
    trajectory_id: Optional[str] = None,
    learner_version: Optional[str] = None,
    exam: bool = True,
    push: Optional[bool] = None,
) -> LessonResult:
    """Learn from one finished campaign: import it into the trajectory
    store, mine it, grade the bank on it (exam-before-lesson), then run
    the update crew — evidence-priced cards, one tagged bank commit.

    Args:
        source: What to learn from —
            - the SolutionResult returned by evolve() (its campaign
              workspace is read directly; the natural closed-loop call),
            - a path to a campaign directory (archived + imported), or
            - a trajectory id already in the store (import skipped).
        trajectory_id: Store id when importing a directory; default is
            derived (<goal-slug>/<UTC-stamp>). Ignored for store ids.
        learner_version: Update-crew version; default from config
            (learning.update_crew.default_version).
        exam: Run the hindcast exam against the pinned pre-lesson bank
            head first (the honest credit assignment). False is for
            development replays only — the operating loop keeps it True.
        push: Push the bank commit to learning.bank.remote. None (the
            default) means "push exactly when a remote is configured".

    Returns:
        LessonResult — what changed in the bank and the paper trail.
    """
```

Semantics, precisely:
- **Dispatch**: `SolutionResult` → its workspace dir; a string that
  resolves to an existing store id → skip import; any other string →
  must be an existing directory (fail loud otherwise) → tar + import.
- **Idempotency**: importing an id the store already holds raises (the
  store's identity contract); re-learning an already-banked trajectory
  is refused with the lesson commit named — never a silent duplicate.
- **The exam pin**: the bank is cloned and its head recorded BEFORE the
  lesson; the exam grades that head. The lesson can never grade itself.
- **No hidden network**: `push=None` behavior is fully determined by
  config; `push=False` always keeps the commit local.

## 3. `LessonResult` — the mirror of `SolutionResult`

```python
@dataclass
class LessonResult:
    """The artifact produced by Kapso.learn(). Not just cards — the
    entire learning attempt, auditable end to end."""
    trajectory_id: str
    bank_head_before: str           # the examined head
    bank_head_after: str            # == before when nothing was admitted
    cards_created: List[str]
    cards_updated: List[str]
    exam_report_path: Optional[str] # None when exam=False
    lesson_report_path: str
    metadata: Dict[str, Any]        # durations, learner_version, pushed

    @property
    def admitted(self) -> bool:     # the peer of SolutionResult.succeeded
        return self.bank_head_after != self.bank_head_before

    def explain(self) -> str: ...   # summary, same idiom as SolutionResult
```

Coherence map — the two results rhyme deliberately:

| `evolve()` → SolutionResult | `learn()` → LessonResult |
|---|---|
| `goal` | `trajectory_id` |
| `code_path` (what was built) | `bank_head_after` + card lists (what was learned) |
| `succeeded` | `admitted` |
| `experiment_logs` | `exam_report_path` + `lesson_report_path` |
| `metadata` (cost, timestamps) | `metadata` (durations, learner_version, pushed) |
| `explain()` | `explain()` |

## 4. The other half of coherence: `evolve()` serves the bank itself

Today serving injection is benchmark-runner code (adoption hook 2);
bare `Kapso.evolve()` never serves. That asymmetry breaks the facade
loop, so it moves into evolve:

- When `learning.serving.enabled` is true and the bank home exists,
  `evolve()` stages serving at launch: `prepare_campaign_serving()`
  against the campaign workspace, the knowledge intro appended to the
  problem context, `bank_serving` threaded into the strategy params
  (the thread already exists end to end since 09302f87 — this is the
  one missing caller).
- Task coordinates for bare goals: `{"family": <mode or "generic">}`;
  a new optional `evolve(serving_scope={...})` kwarg lets callers with
  real coordinates pass them. Benchmarks keep their own richer hooks
  unchanged.
- Serving off, or no bank yet → exactly today's behavior, byte for byte.

With that, the closed loop is three lines of user code:

```python
k = Kapso()
solution = k.evolve(goal=goal, eval_dir="./eval", time_budget_minutes=240)
lesson   = k.learn(solution)          # import → mine → exam → lesson → push
print(lesson.explain())               # cards, heads, admitted
next_sol = k.evolve(goal=next_goal)   # served the updated bank
```

## 5. What deliberately stays OUT of the facade (Rule 10)

Operator machinery remains CLI-only: `learn grade --split` /
`develop` / `gauntlet` (the development regime), `codify` (GCP replay
flips), `behave` (semantic suite). They run the learning SYSTEM's own
lifecycle, not the user's loop. A future `kapso.certify(pairs)` for A/B
promotion is noted and deferred — it needs fleet orchestration the
facade should not own today.

## 6. Config addition (Rule 1)

One new key, sourced not invented: `learning.update_crew.
default_version` (the crew version `learn()` uses when the kwarg is
omitted — the CLI keeps its explicit `--learner-version`). No other
knobs: everything else `learn()` needs already lives under `learning:`.

## 7. Implementation map (small; the chain already exists)

`kapso.py`: rename `learn` → `learn_knowledge`; new `learn()` composing
existing pieces (`import_archive` / `TrajectoryStore` / `MiningFrame` /
`GradingFrame.grade_exam` / `UpdateFrame.run_update` — the same calls
`cmd_learn ingest` makes, plus bank-head bookkeeping for LessonResult);
`LessonResult` in `execution/solution.py`'s idiom (new
`learning/lesson_result.py`); the evolve serving hook (§4) in
`kapso.py`'s evolve path; config key; class docstring rewrite; tests —
facade-level: dispatch (result/path/id), idempotent-refusal, exam-pin
(head recorded before lesson), LessonResult.admitted, evolve-serving
staging when enabled + byte-identical problem context when disabled.
CLI stays as is (it is the same chain spelled out).

---

# §8. The memory model — connecting the two learners

The facade now carries two learning systems with different backends:

| | `learn_knowledge()` — knowledge | `learn()` — experience |
|---|---|---|
| source | external: repos, research outputs, ideas | internal: the agent's own campaigns |
| content | wiki pages / workflow repos | evidence-priced cards |
| backend | wiki dir + KG backends + `.index` pointer | bank git repo (home + optional remote) |
| trust model | curated, unpriced | measured, reliability-scored |
| evolve read path | knowledge_search (KG gates) + workflow-repo search | serving (intro + bank tools) |

## 8.1 Unification options considered

- **A — one backend** (migrate wikis into the bank as a card species, or
  cards into the KG): rejected. The trust models are incompatible (a
  curated page cannot carry an evidence ledger; pricing machinery is
  meaningless for imported docs), the read modes differ (semantic search
  vs agentic index), and the migration buys adapters, not deletions
  (Rule 10).
- **B — one facade model, two stores** (chosen): the agent has ONE
  memory with two stores — **knowledge** (what others know, imported)
  and **experience** (what it measured by doing). Unification lives in
  the mental model, the constructor, the status surface, and evolve's
  automatic consumption of BOTH — never in the backends.
- **C — one verb dispatching by source type**: rejected in §1 (a pun,
  not a unification); the connection problem is state and read paths,
  not naming.

## 8.2 The connection design

**Constructor — symmetric, config-defaulted, override-able:**

```python
Kapso(
    config_path=None,
    kg_index=None,   # knowledge store connection (as today)
    bank=None,       # experience store home; default:
                     #   config learning.bank.local_path
)
```

Both stores resolve ONCE at construction into an internal memory
descriptor; `learn()`, `learn_knowledge()`, and `evolve()` all read that
resolution — never re-reading config at call time, and no per-call
store overrides (one resolution point; the CLI keeps `--config`).

**Status surface — one place to ask "what does this agent know":**

```python
kapso.memory            # MemoryStatus
#   .knowledge:  index path | None, backend type, enabled
#   .experience: bank path, head, active cards, store trajectories,
#                serving_enabled
kapso.memory.explain()  # one readable summary
```

**Provenance — every result stamps what it drew on:** evolve stamps
`metadata["kg_index"]` and `metadata["bank_head_served"]` into
SolutionResult; learn already stamps heads into LessonResult. A solution
is always traceable to the exact memory state that produced it.

**Read paths stay SEPARATE in the prompt, deliberately.** Knowledge
arrives through the existing KG gates and workflow-repo search;
experience through the serving intro + bank tools. They are different
epistemic classes — one curated, one measured — and merging them into a
single retrieval list would launder that distinction. The ideation
prompt already frames each correctly (INDEX.md-style bank text vs "cards
are measured practice").

**One repair the simulations forced (S1):** `learn_knowledge()` merges
into the KG backends but never refreshes `self.knowledge_search` — a
Kapso built without `kg_index` stays null-search, so a same-object
evolve is blind to knowledge it just learned (verified at
kapso.py:learn tail — pipeline.run() then return, no search rebuild).
Fix, part of this design: after a merge (`skip_merge=False`),
`learn_knowledge()` initializes/refreshes `self.knowledge_search` from
the config preset exactly as `index_kg()` already does post-index.

## 8.3 The permutation simulations

| # | sequence | outcome under this design |
|---|---|---|
| S1 | `learn_knowledge(repo)` → `evolve` | WORKS after the 8.2 repair (was: silently blind). evolve consults the fresh KG; bank empty → serving intro honestly reports gaps (or serving off → today's behavior). |
| S2 | `evolve` → `learn(sol)` → `evolve` | The §2-§4 loop: second evolve's launch-time bank clone sees the new head. |
| S3 | `learn_knowledge(repo)` → `evolve` → `learn(sol)` → `evolve` | Both stores feed the last evolve through their separate slots; SolutionResult stamps kg_index + bank_head_served. |
| S4 | `evolve` → `learn(sol)` → `learn_knowledge(research_out)` → `evolve` | Order-free: `research()` output feeds learn_knowledge (already supported); stores are independent, so interleaving cannot corrupt either. |
| S5 | two projects | `Kapso(bank="./proj-a-bank")` vs `Kapso(bank="./proj-b-bank")`; kg likewise per `kg_index`. Concurrent READS of one bank are safe (serving pins per-campaign clones). Concurrent `learn()` into one bank is NOT supported — the update transaction assumes a serial writer; documented contract: one learn at a time per bank home. |
| S6 | cross-process resume | `learn("path/to/campaign")` and `learn("store/id")` cover results from other processes; nothing depends on in-object state. |
| S7 | cold start | no kg, no bank: evolve is byte-identical to today. First `learn()` requires the bank home (init_bank or config-created); error names the fix. |

## 8.4 Facade summary after §§1-8

```python
k = Kapso(kg_index="data/indexes/ml.index")        # knowledge connected
k.learn_knowledge(Source.Repo(url), k.research(q)) # imported knowledge
sol   = k.evolve(goal=..., time_budget_minutes=240) # consults BOTH stores
les   = k.learn(sol)                                # experience earned
print(k.memory.explain())                           # one status view
sol2  = k.evolve(goal=...)                          # smarter on both axes
```

Implementation adds to §7's map: the `bank=` constructor arg + memory
resolution, `MemoryStatus` + `explain()`, the S1 refresh in
learn_knowledge, provenance stamps in evolve, and facade tests for
S1-S4 (S1's regression test: null-search Kapso + learn_knowledge →
evolve consults the KG).
