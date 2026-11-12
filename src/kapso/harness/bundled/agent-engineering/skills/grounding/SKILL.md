---
name: grounding
description: How a scaffold finds what a task needs in a repository, a document base or an API, with the strategy per source kind, whole-unit reads, reference following, stopping rules, conflict resolution by authority and applicability, the measured limits of retrieval and when a trained retriever or a retrieval subagent earns its place. Use when building the ground phase, or when the triage shows a procedure never found, a wrong target, or a condition missed.
---
# Grounding

## What the measurements say

- Retrieval is the ceiling. On one document-grounded task, handing the model the documents
  it needed raised frontier pass rates by 13 to 15 points over any search setting;
  document recall under search sat between 45 and 62 percent; the whole corpus in the
  window halved the pass rate.
- Terminal-style search (list, grep, read files) beat dense and sparse retrieval on
  average, and one shipped coding harness dropped its index for grep because agentic search
  "outperformed everything". A reranker added nothing; grep alone was no better than
  retrieval; retrieving 20 instead of 10 changed nothing.
- The counter-evidence is also measured: a retriever trained on the codebase gained 12.5
  percent answer accuracy for one product, and another found trajectories spending more
  than 60 percent of their first turn retrieving context and trained a retrieval subagent
  rewarded for precision, because polluting the main context cost more than missing some.
  Measure on your task; keep grep regardless.
- Frontier agents run 9 to 20 searches per task, and twice the searches did not buy a
  higher pass rate. The cost is in reading and acting, not in querying.

## Strategy by source kind

- **A repository.** Retrieve the whole changed unit and the callers or tests that
  constrain its behaviour, and preserve those constraints through context reduction;
  search by symbol and by the ticket's own words; follow imports before editing. The
  tests are the specification: a failing test names the behaviour, a passing one names
  the contract that must survive. Knowing what is installed before the loop starts is
  the harness's job: a quarter of command failures on one terminal benchmark were
  missing executables.
- **A document base.** Establish the index and each source's authority. Retrieve the
  governing unit and its referenced prerequisites, exceptions and definitions; read
  enough of each unit to preserve the conditions for the pending decision, since a
  snippet loses the condition that matters. Stop searching when those fields are sourced;
  otherwise search for the specific missing field by name, not for the operation again.
- **An API or a schema.** Read the schema before the first call; treat an error message as
  documentation; one probing call with a harmless argument beats three guesses.

## Stopping rules that save an episode

- Stop when the compiled plan (action-review) has a source for every field.
- A search that returns what the last search returned is a signal to read, not to
  rephrase.
- Do not invent a tool, a function or a document that nothing mentions; use the runtime's
  discovery mechanism, and distinguish an unsuccessful search from proof of absence.
- When two sources disagree, resolve by the task's authority, version and applicability
  rules; a consequential conflict those rules do not settle stays in the working state as
  unresolved.
- A question to the user or the ticket's author is cheaper than a tenth search when the
  missing field is theirs to give.

## Record ids, keep the spans that decide

The working state keeps the ids, paths and the exact spans that support unresolved
decisions; the rest of the material leaves the window (context-policy). Re-read a source
when it changes, when a condition is missing, or when a decision needs detail that was
discarded.
