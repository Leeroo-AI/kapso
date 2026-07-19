# M4 — GenericSearch and experiment bridge

Parent plan: [`00-orchestrator-plan.md`](00-orchestrator-plan.md)

Depends on: M1, M2, and M3.

## Objective

Connect the durable candidate pipeline to the existing GenericSearch iteration
without changing the responsibilities of implementation, integrity checking,
feedback, fidelity, or evaluation.

## Owned responsibilities

- `IdeationEngine` transaction orchestration.
- GenericSearch strict replacement routing.
- Per-operator parent snapshot resolution and read-only materialization.
- Selected-idea to SearchNode bridge.
- Idea and selection-batch links on nodes.
- Idempotent node creation after selection.
- Existing agent telemetry aggregation across the new pipeline.

## Proposed code surface

```text
generic/ideation/
  engine.py

generic/strategy.py
execution/search_strategies/base.py
execution/search_strategies/strategies.yaml  # only if configuration lives here

tests/
  test_ideation_engine.py
  test_ideation_parent_bridge.py
  test_ideation_v3_strategy.py
```

## Integration tasks

- [x] Initialize `IdeaArchive` and `IdeationEngine` from the GenericSearch
      workspace and configuration.
- [x] Construct Codex and Claude runner configurations through the existing CLI
      auth, environment-hygiene, timeout, and telemetry paths.
- [x] Keep `OPENAI_API_KEY` in the orchestrator process for embeddings and out
      of both coding-agent subprocess environments.
- [x] Construct `IdeationCapacityView` only from the existing budget snapshot,
      fidelity decision, and fidelity timing authority; add no local duration
      or reserve estimate.
- [x] Replace `_generate_solution` with the evidence, directive, archive,
      candidate, and selector stages.
- [x] Persist `PLANNED`, `GENERATED`, `ANALYZED`, and `SELECTED` boundaries in
      order.
- [x] Return a typed selected idea and immutable parent snapshot rather than a
      bare string.
- [x] Add required `idea_id` and `selection_batch_id` to Generic `SearchNode`
      projections and update
      every constructor and serialized fixture to the new shape.
- [x] Copy selected proposal text to `SearchNode.solution`.
- [x] Link the node idempotently before implementation begins.
- [x] Preserve current phase telemetry and include generator, repair, analyzer,
      and selector costs in ideation totals.

## Parent and worktree tasks

- [x] Refactor current `_select_parent` resolution so existing `best` and
      `baseline` behavior remain available as parent plans.
- [x] Add `SPECIFIC_EXPERIMENT` and `RECOVER_BRANCH` resolution.
- [x] Resolve branch, node, and Git ref once and validate their agreement.
- [x] Materialize one read-only worktree per distinct parent ref needed by the
      candidate batch.
- [x] Reuse views for members sharing a ref and clean every view on success or
      failure.
- [x] Carry the selected snapshot unchanged into implementation, diff, and
      feedback base-ref arguments.
- [x] Reject a stale/missing selected ref instead of falling back to `main`.

## Existing lifecycle preservation

The bridge must hand off to the current code at the same semantic point as the
old solution string. These behaviors must remain unchanged:

- fidelity `VALIDATE` short-circuits ideation entirely;
- implementation uses the existing coding-agent path;
- integrity gates run before accepting score/feedback;
- evaluation attempts remain append-only;
- node telemetry is stamped once;
- `node_history` remains contiguous and canonical; and
- the orchestrator receives a normal `SearchNode`.

## Tests

- V3 selected idea is persisted before SearchNode creation.
- Crash after selection resumes through one idempotent node link.
- Best, baseline, specific, and recovery plans materialize the expected ref.
- Multiple candidate parents do not change the root checkout.
- Node lineage, implementation base, diff base, and feedback base agree.
- `VALIDATE` performs no ideation or idea-archive mutation.
- V3 node round-trips through existing serialization with idea links.
- Ideation phase cost equals all v3 subphase costs without double counting.
- Generator and selector roles can independently use Codex CLI or Claude Code.
- A subprocess-environment test proves the embedding key is absent from both
  CLI runners.

## Definition of done

- A mocked v3 iteration reaches existing implementation with a linked node.
- Superseded ideation tests are rewritten around the v3 contract rather than
  preserving the old call order.
- No experiment-memory writes are moved into GenericSearch.
- All parent-ref safety tests pass under failures and cleanup.

## Non-goals

- Final experiment projection or MCP rendering.
- Cross-store outcome reconciliation.
- Final deletion of now-unreachable legacy helpers, owned by M6.
