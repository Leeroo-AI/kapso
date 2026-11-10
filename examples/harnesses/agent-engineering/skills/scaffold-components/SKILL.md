---
name: scaffold-components
description: The catalogue of scaffold components with the measured effect of each under its conditions, the observed failure each one answers and the evidence needed to keep it, the expert's build order, what is known not to work, which disagreements in the literature are model-vintage disagreements, the ledger entry every component carries so it can be retired, and the replay and mechanism checks that show a component works for the reason claimed. Use when proposing a component in ideation, building one in implementation, or deciding what to remove.
---
# Scaffold components

A scaffold supplies the model's runtime interface, state and control flow. Its components
encode assumptions about what the model cannot do, and those assumptions expire: context
resets, sprint decomposition, think tools, unified-diff formats and ALL-CAPS emphasis were
each essential for one model generation and dead weight for the next, and removing all of
them at once hid which ones were load-bearing. Build in order, measure one at a time, and
write the expiry condition with the component.

## The build order

1. The completion check the scaffold runs and the specification it gives the model
   (done-check-and-eval), with the supplied evaluation contract read, never edited.
2. A shell tool plus an exact-match editor, the loop, its caps (tool-layer,
   deterministic-guards). A bash tool and a string-replace editor took a 2024 model to the
   state of the art on its own.
3. The sandbox and a few deny rules (deterministic-guards).
4. Cache-stable, append-only context plumbing (context-policy, grounding).
5. The model-facing scaffolds: planners, review steps, best-of-n, memory, phase-dependent
   thinking. Last, because they are the ones that expire.

## Choose a component from an observed failure

| Observed failure | Intervention to test | Evidence needed to keep it |
| --- | --- | --- |
| stops with an observable requirement unmet | a completion gate on the checks the scaffold may run | fewer premature stops without blocked correct completions |
| right intention, invalid tool call | parser repair, a grammar on the arguments, a clearer tool contract | fewer invalid calls, no guessed argument executed |
| long observations bury the needed evidence | filtering in code, selective reads, compaction at boundaries | lower cost per episode with the decision evidence retained |
| a consequential action violates a prerequisite | a deterministic precondition gate; a model review only for the residual ambiguity | fewer invalid actions, with the false rejections measured |
| repeats actions without state progress | state-aware loop detection and bounded recovery | less wasted work without cutting necessary repeats |
| forgets obligations over a long horizon | a sourced obligation ledger in the working state | fewer omissions at an acceptable update cost |
| several candidates, poor selection | an execution-grounded selector | better selected outcomes at the same generation budget |

Published effects motivate these interventions; they do not predict their size on another
checkpoint or task family. Reject a component when its triggering failure is absent.

## Components by measured effect, with their conditions

| Component | Measured effect | Conditions |
| --- | --- | --- |
| Specification: requirements, interface, rules, examples of should and should-not | 8.4 to 25.9 on one hard coding benchmark for one frontier model from requirements text; 33.2 to 10.8 on one support benchmark when the policy text was removed; over 60 percent of failed tasks in one audit violated an explicit instruction | the largest lever in those measurements; the specification is the scaffold's, the task statement is the task's |
| Execution-grounded verification: real tests, reproduction scripts, a read-back of the state before stopping | +8 to +11 with real tests, +5 from reproduction tests, +15.6 from an objective verification step, on coding benchmarks; agents that verified were 13 to 87 percent likelier to succeed in one trace study | only with an external signal; self-reflection alone regressed |
| Best-of-N plus a selector | +3 to +8 at 5 to 50 times the cost on coding benchmarks; selection, not coverage, is the ceiling (69.8 covered, 57.4 selected) | at submissions and irreversible actions; the selector is where the work goes |
| The agent-computer interface: an editor with a unique exact match and lint-on-edit, a bounded file viewer, summarized search, collapsed old observations | −7.7 without an edit tool; +3.0 lint; +3.7 and +5.3 viewer; +6.0 search; +3.0 history, all on one 2024 model | model-specific formats expire; the strict exact-match contract became viable as models stopped needing fuzzy matching |
| Tool-use examples; deferred tool loading; few distinct tools | parameter accuracy 72 to 90 with examples in one study; context cut 85 percent and accuracy 49 to 74 with deferral in another; many distinct tools worked where few overlapping ones failed | tool-layer |
| Data filtered in code before the model sees it | 150,000 tokens to 2,000 in one deployment; 45.3 to 61.6 on one browsing benchmark when the model filtered its own tool output | context-policy; needs a sandbox |
| Curated memory and strategy notes | +3 to +8 across three studies; raw or uncurated memory scored below no memory | only when the loop writes and reads it as a phase, and curates it |
| Observation and action space simplification | +11.6 action, +4.7 observation; an accessibility tree 83.1 against pixels 48.9, on browsing benchmarks | the observation space is the harness |
| Ledger or planner on long horizons | −31 percent without a ledger orchestrator on one 2024 model; +7.4 from a planning tree; nothing on a support benchmark for function-calling models; a 2026 study still found large gaps on long tasks | horizon-dependent; write the expiry condition |
| Reasoning effort | +14 on one knowledge-heavy benchmark from none to high; equal or worse in 21 of 36 paired runs elsewhere | tune per task when the task statement permits it, never default |
| Tree search | +5 to +7 for weaker models at 1.6 to 5 times the cost | measured on weaker models; a cost line, not a free gain |
| Multi-agent | +90 percent on breadth-first research at 15 times the tokens; −0.2 to −1.0 from delegation elsewhere; role splits spent more on coordination than work | context isolation, breadth, black-box verification only |
| Automated harness search | +1.2 to +11.8 per step on the family it tunes for; transfer at most 2.9 | per-domain, against a sealed eval |

## Known not to work, or not to matter

- Self-correction without an external signal (95.5 to 89.0 on one benchmark over two
  rounds); a think or planning tool for native function-calling models; interleaved
  reasoning text over function calling; role-split multi-agent teams; a separate
  fix-regressions agent.
- Prompt emphasis (it makes rules over-applied, not obeyed); long always-on instruction
  files (adherence drops); "keep going, don't be lazy" (now causes over-work); describing
  the repository's architecture instead of letting the model read it.
- More searches or tool calls by themselves; rerankers on retrieval, grep alone, retrieving
  20 instead of 10, the whole corpus in the window; lowering the temperature to look
  stable.

## Vintage: which disagreements dissolve with the model generation

Single thread against multi-agent, context resets against compaction, small step caps,
sprint decomposition, prompt emphasis, think tools, subagents as context savers, edit
formats: each was right for one model and wrong for the next. Durable across vintages:
the attention budget, compaction being irreversible, the verifier outside the model, the
guards in code, bounded loops, the sealed eval.

## The ledger

Every component carries an entry in `results/ledger.json`, and the attempt notes repeat it:

```json
{"component": "pre_commit_review", "phase": "act", "model": "<name and version>",
 "failure_class": "argument_differs", "hypothesis": "the model copies ids from memory; a review step traces them",
 "measurement": {"before": "28/60", "after": "35/60", "tasks": 20, "trials": 3},
 "retire_when": "the baseline model makes fewer than 2 argument errors in 60 episodes"}
```

Components are removed one at a time, each removal measured; "radical simplification"
failed where it was tried because it hid what was load-bearing. When a new component's
failure class or build-order axis overlaps an incumbent's, the attempt also re-ablates that
incumbent in a single-flag paired run, the incumbent off and the new component and the rest
on, to see whether the new one subsumed it; retirement is measured this way, one flag at a
time, never assumed from the overlap.

## How to test a component

One module or one function per component, named for the phase it serves; a flag that
turns it off, so the ablation is one setting; a transcript line when it runs. First
replay a permitted trace prefix that ends just before the target failure, in an isolated
or read-only environment, and compare the next decision with and without the component;
then run complete paired episodes to see downstream costs and regressions. Replay never
repeats a live external write. For a component claimed to work by a general mechanism,
construct public synthetic cases that keep the operation but rename entities and tool
identifiers, vary amounts and reorder irrelevant documents, and change one governing
condition to see the decision change; a failure here says the component depends on
recognition, not on the mechanism. Trigger this check whenever a measured gain did not pass
through the episodes the hypothesis named, or whenever a component's effect cannot be
traced to its stated mechanism: until the synthetic cases confirm it the mechanism is
unconfirmed, and the attempt is not made a parent.
