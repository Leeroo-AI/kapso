# Onboarding E2E findings — 2026-08-27

Issues surfaced by running the complete production loop from the
**published PyPI package in a fresh isolated environment** — the way an
outside engineer meets Kapso, not the way we run it from the source tree.

Run setup: fresh Python 3.13 venv, `pip install leeroo-kapso` (0.3.3 at
start, 0.3.4 after the first fix shipped), task = the ML-model-development
example (synthetic Spaceship-Titanic data), loop = light `research()` →
`learn_knowledge()` → `evolve()` → `learn()` → `evolve()` with serving
enabled and the learning homes sandboxed into the run dir. This is a
living document for that run — findings from later stages get appended.

Status legend: **SHIPPED** = fixed and released; **OPEN** = documented
here, awaiting a decision (framework changes need explicit approval).

---

## 1. `.env` was never read from the user's directory in pip installs — SHIPPED (0.3.4)

**Symptom.** Following the README exactly (`echo 'OPENAI_API_KEY=sk-...'
>> .env`, then `kapso doctor`) failed the OPENAI_API_KEY check even with
the `.env` sitting in the current directory.

**Mechanism.** `cli.py` and `kapso.py` called `load_dotenv()` with no
argument. python-dotenv's no-arg default anchors `find_dotenv()` at the
*calling module's file* and walks up from there — in a pip install that
is `site-packages`, whose parent chain never contains the user's project
directory. Every dev checkout masked the bug: with CWD = repo root, the
repo's own `.env` happens to sit on the walk-up path.

**Fix.** `load_dotenv(find_dotenv(usecwd=True))` in both entry modules —
resolve from the process CWD upward, which is what a CLI user expects.
Commits `9ac8202e` + release `e1085071`, live as **leeroo-kapso 0.3.4**;
re-verified from a clean venv (doctor fully green, and the run's spawned
sessions inherit the `.env` credentials).

## 2. Packaged bank defaults are our production values — SHIPPED (0.3.5)

**Resolution (52288da9), redesigned git-native per review:** the
`learning.bank.remote` config key is deleted — a bank carries its share
remote as its own `origin`, exactly like any git repo. Packaged default
is the neutral `data/kapso-bank.git`, `learn()` pushes to `origin` only
when one is attached, and attaching is one command: `kapso bank connect
<url>` (or `kapso bank create org/name`, via gh). A push destination is
also **preflighted with ls-remote at learn() start**, so auth failures
cost seconds, never hours. The production relbench bank home was renamed
in place; its already-set origin carries the relbench identity — no
overlay needed anywhere. Original finding kept below.

### Original finding

The config shipped in the wheel carries:

```yaml
learning:
  bank:
    local_path: data/kapso-bank-relbench.git
    remote: https://github.com/Leeroo-AI/kapso-bank-relbench.git
```

Two consequences for an outside pip user:

- `kapso learn init-bank` on the default config creates a bank named
  after *our relbench campaign* in their project.
- `Kapso.learn()` pushes whenever a remote is configured
  (`should_push = bool(remote)` when `push` is not passed) — so their
  first `learn()` runs the full multi-hour mining/grading/update
  pipeline and then **crashes at the very end** trying to push to a
  private Leeroo repo they cannot auth against. Worst possible failure
  point: all the work done, nothing banked-and-pushed cleanly.

**Recommendation.** Packaged default becomes `local_path:
data/kapso-bank.git` + `remote: null`; the relbench bank moves to a
production overlay (mode config or explicit `--config`) on our side.
Not applied — it touches the production paths this branch's live
relbench learning runs read.

## 3. README Basic Usage omits the bank-init step — SHIPPED (0.3.5)

**Resolution (52288da9):** option (b) — `learn()` founds a missing bank
home automatically, deleting the setup step from the golden path; the
README documents the auto-creation and the share command. Original
finding kept below.

### Original finding

The quickstart's core loop (`evolve → learn → evolve`) fails at
`learn()` for a fresh user: the bank home does not exist yet.
The error is *guided* — `FileNotFoundError: bank home ... does not
exist — run 'kapso learn init-bank' (or init_bank()) first` — so the
user can recover, but the README snippet as written does not run to
completion, and the recovery command currently mints the relbench-named
bank (finding 2).

**Recommendation.** Either (a) one added line in the README before the
`learn()` call (`kapso learn init-bank`), or (b) auto-init an absent
bank home on first `learn()` — initialization of an empty configured
path is unambiguous, and the explicit-init requirement mainly protects
against typo'd paths, which the guided error already handles. Decide
(a) vs (b); (b) deletes a setup step from the golden path.

## 4. Live suites in `tests/` look hermetic — SHIPPED (0.3.5)

**Resolution (29255ffc):** `tests/conftest.py` registers a `live` marker
and a `--run-live` opt-in; the four suites carry a module `pytestmark`
and skip in ~2s without the flag (verified for direct file invocation
too). Note kept open inside the marker comments: the mining/grading
frame suites are fake-boundary by design yet a full run empirically
spawned a real claude session — the leak is still unfound. Original
finding kept below.

### Original finding

`tests/test_mining_frame.py` and `tests/test_grading_frame.py` spawn
**real claude crew sessions** when run (same family as
`test_researcher_modes.py` / `test_research_ingestors.py`, which spawn
real codex sessions). There is no marker, naming convention, or skip
gate separating them from the hermetic suites. Cost of the trap,
measured today: a gate run padded with these files hung pytest for 10
minutes while silently burning real subscription quota mid-CI. An
outside contributor running a plain `pytest tests/` hits the same
thing (plus the infra-dependent suites that hang collection without
Weaviate/Neo4j).

**Recommendation.** An explicit opt-in for live suites — e.g. a
`--run-live` pytest flag (registered in `conftest.py`, suites skip
without it), so `pytest tests/` is safe-by-default for contributors.
Env-var gating is ruled out by the no-env-config rule.

## 5. Config names models an account may not be able to serve — SHIPPED (0.3.6)

**Resolution (99027d2c):** both halves — the README's "Choosing models"
section documents the copy-packaged-config override pattern, and
`kapso doctor --models [--config X]` live-probes every distinct
{cli, model} pair the active config names with a one-token call
(claude `-p`; codex `exec --sandbox read-only --skip-git-repo-check`,
stdin closed). A capped model fails the doctor in seconds with the
CLI's own message. Original finding kept below.

### Original finding

The packaged config hard-names `claude-fable-5` across the learning
stack (mining lead/flow-writer/critic, grading verifier, update-crew
lead/critic, codify judge, behavior reviewer). Model access is
account-dependent: this run's designated subscription had its Fable-5
window capped (the CLI returns "You've reached your Fable 5 limit"
while opus-5 works fine on the same account). Nothing preflights this —
the failure would surface as crew-session errors deep inside `learn()`.

What worked, and is the pattern worth documenting: the user-side
override after `pip install` — load the packaged default
(`kapso.kapso.DEFAULT_CONFIG_PATH`), edit the dict (this run swapped
all 8 `claude-fable-5` entries under `learning:` to `claude-opus-5`),
write it next to your project, pass `Kapso(config_path=...)`. The E2E
driver (`/home/ubuntu/kapso-onboarding-test/full_loop.py`) is a working
reference.

**Recommendation.** (a) Document the override pattern in the README /
docs quickstart; (b) optionally teach `kapso doctor` to probe each
distinct `{cli, model}` pair in the active config with a one-token
session, so a capped or unavailable model is caught before a run, not
mid-`learn()`.

---

# Run record — LOOP CLOSED, all stages green

The full loop completed 2026-08-27 17:55 UTC, driver exit 0, on
**leeroo-kapso 0.3.4 from PyPI** in a fresh Python 3.13 venv.

| Stage | Wall time | Result |
|---|---|---|
| research (light, idea) | 6 min | 16.8KB findings, one codex session |
| learn_knowledge | 1h51m | **53 pages created**, 0 errors, merged into KG |
| evolve #1 | 23 min (of 45 budget) | **0.89 accuracy** (target 0.78, baseline ~0.50), succeeded, stopped early; served the founding bank head `e360a54b` |
| learn | 4h35m | **admitted; 10 cards created**, 0 updated; bank `e360a54b → 2e6a58a8`; harvest 33m / mine 33m / exam 11m / lesson 3h51m |
| evolve #2 | 27 min | **0.89 accuracy**, succeeded; **served the post-lesson head `2e6a58a8`** |
| loop-closure check | — | **PASSED**: evolve #2's served head == lesson's `bank_head_after` |

Total: **7h21m** end to end. What this validates: the published wheel is
complete (crew prompts, configs, entry points), Python 3.13 works, CWD
`.env` credential flow works (finding 1's fix), the post-install config
override path works (finding 5's pattern, including the model swap),
serving hands each campaign the exact bank head it should see, and the
experience loop genuinely closes — the second campaign consumed the
cards the first one earned. Both campaigns saturated the accuracy target,
so this run demonstrates loop *mechanics*, not served-cards *uplift* —
uplift needs a task the first campaign does not saturate.

## 6. Lesson-phase margin vs the crew timeout — OPEN (observation)

The single-trajectory founding docket consumed 3h51m of the 240-min
update-crew cap — a 9-minute margin. The cap was calibrated for the
fable-5 crews; this run's opus-5 swap (finding 5) nearly exhausted it.
A second trajectory in the docket, or a slightly chattier session,
would have tripped the cap and failed `learn()` at the very end.
**Recommendation:** consider the timeout a per-model knob (or scale
with docket size), and say in docs that model swaps change pacing.

## 7. "Light" research does not bound ingestion — OPEN (observation)

The fast-finish lever (`depth="light"`, narrow question) bounded the
research stage (6 min) but not extraction: 16.8KB of findings became
53 wiki pages and 1h51m of `learn_knowledge`. Thorough extraction is
by design — but the quickstart's mental model ("simple research to
finish fast") should warn that ingest time scales with extractable
substance, not with research depth.

## 8. Unclosed SSL socket warning at driver exit — OPEN (cosmetic)

Interpreter teardown prints `ResourceWarning: unclosed <ssl.SSLSocket
...>` after a run that used the KG backends — a client (Weaviate/Neo4j)
is not closed on the facade path. Harmless, but it is the last line a
new user sees after their first successful run. **Recommendation:**
close KG clients when the facade finishes with them.
