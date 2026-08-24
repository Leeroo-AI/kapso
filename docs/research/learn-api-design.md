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
