# M1 — domain model and idea archive

Parent plan: [`00-orchestrator-plan.md`](00-orchestrator-plan.md)

## Objective

Create the stable, strategy-local domain and persistence substrate on which all
other ideation-v3 modules depend. This module contains no model or coding-agent
calls, policy judgment, Git operations, or experiment execution.

## Owned responsibilities

- Frozen dataclasses/enums and JSON schemas.
- Stable campaign, batch, idea, claim, and gap identifiers.
- Legal lifecycle transitions.
- Atomic, strict, campaign-local `IdeaArchive` persistence.
- Idempotent batch, selection, link, and outcome mutations.
- Query primitives used by evidence and retrieval modules.
- One current persisted shape with no migration hooks.

## Proposed code surface

```text
src/kapso/execution/search_strategies/generic/ideation/
  __init__.py
  types.py
  archive.py

tests/
  test_ideation_domain.py
  test_idea_archive.py
```

Default archive path:

```text
<workspace>/.kapso/idea_archive.json
```

## Contract tasks

- [x] Define string enums for idea, batch, gap, claim, policy, operator, and
      parent-plan states.
- [x] Define immutable value objects listed in the orchestrator contract
      freeze.
- [x] Define `CodingAgentCallRequest` and `CodingAgentCallResult` without
      importing a concrete CLI adapter.
- [x] Define embedding records and telemetry with provider, model, dimensions,
      input hash, vector, call count, input tokens, and duration; never include
      credentials.
- [x] Separate `origin_batch_id` from `selected_in_batch_id`.
- [x] Represent raw score separately from normalized utility.
- [x] Require source references for non-insufficient evidence claims.
- [x] Encode idea-to-node linkage without importing `SearchNode`.
- [x] Make `to_dict`/`from_dict` reject unknown and missing fields explicitly.
- [x] Validate finite numeric values and reject booleans as integers.
- [x] Define transition tables and reject illegal backward transitions.

## Archive operations

The archive interface should be narrow and idempotent:

```text
create_batch(batch) -> IdeaBatch
add_ideas(batch_id, ideas) -> archive_revision
record_analysis(batch_id, analysis) -> archive_revision
record_selection(batch_id, decision) -> archive_revision
link_experiment(idea_id, node_id, selection_batch_id) -> archive_revision
record_outcome(idea_id, outcome) -> archive_revision
abandon_batch(batch_id, reason) -> archive_revision
get_batch(batch_id)
get_idea(idea_id)
list_recent_ideas(...)
searchable_ideas(...)
list_gaps(...)
```

Mutations accept an expected archive revision where concurrent or replayed
writes could conflict. Repeating the same mutation returns success; repeating
it with different data fails loudly.

## Persistence rules

- [x] Write a complete new document to a same-directory temporary file.
- [x] Flush and atomically replace the prior archive.
- [x] Never leave an empty/truncated archive after interruption.
- [x] Increment a monotonic archive revision per committed mutation.
- [x] Include campaign identity, revision, created/updated timestamps, batches,
      ideas, claims, and gaps.
- [x] Refuse any incompatible persisted shape.
- [x] Avoid storing model transcripts unless explicitly required for audit;
      structured outputs and compact provenance are the contract.

## Tests

- Round-trip every type through JSON.
- Reject invalid enums, IDs, timestamps, transitions, and non-finite numbers.
- Prove duplicate replay of each mutation is idempotent.
- Prove conflicting replay fails.
- Inject failure before atomic replace and verify the previous archive remains.
- Load an empty campaign and a large multi-batch campaign.
- Verify one idea cannot link to two experiment nodes.
- Verify a deferred idea can be selected in a later batch without changing its
  origin batch.

## Definition of done

- Shared contracts are reviewed and marked frozen in the orchestrator plan.
- Archive tests require no agent, Git repository, MCP process, or orchestrator.
- All dependent modules can use the types without importing `strategy.py`.
- Corruption and conflict errors are typed and actionable.

## Non-goals

- Embedding generation or similarity scoring.
- Candidate generation or selection.
- Experiment-store or checkpoint compatibility.
- Choosing the archive's long-term database backend.
