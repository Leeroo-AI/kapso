# M3 — candidate generation, analysis, and selection

Parent plan: [`00-orchestrator-plan.md`](00-orchestrator-plan.md)

Depends on: M1 and M2.

## Objective

Turn a frozen evidence snapshot and `SearchDirective` into a persisted,
structured `SelectionDecision` without executing an experiment.

## Owned responsibilities

- Existing Claude Code and Codex CLI adapters behind one interface.
- Operator-specific structured prompts.
- Independent ensemble generation.
- Prior-idea resurfacing into the considered pool.
- Schema, provenance, duplicate, descriptor, evidence, and capacity analysis.
- One bounded diversity-repair round.
- Selector hard rules, coding-agent criticism, structured fallback order.
- Narrow OpenAI embedding provider and local cosine similarity.

## Proposed code surface

```text
generic/ideation/
  generator.py
  embeddings.py
  analyzer.py
  selector.py

generic/prompts/
  ideation_v3_candidate.md
  ideation_v3_selector.md

tests/
  test_ideation_generator.py
  test_ideation_analyzer.py
  test_ideation_selector.py
```

## Generator tasks

- [ ] Define one `CodingAgentCallRunner` contract implemented by Codex CLI and
      Claude Code runners.
- [ ] Prohibit direct generative calls through `LLMBackend`, LiteLLM, or an
      OpenAI/Anthropic model API inside the candidate pipeline.
- [ ] Reuse current read-only tool restrictions, timeout/cost telemetry, and
      output-salvage behavior.
- [ ] Push the mandatory evidence packet into every member prompt.
- [ ] Assign one operator brief and frozen parent snapshot per member.
- [ ] Keep members independent until all proposals are collected.
- [ ] Parse structured candidates without treating arbitrary full output as a
      valid proposal.
- [ ] Persist every returned candidate, including invalid candidates with raw
      parse provenance.
- [ ] Preserve partial-member failure without hiding quota shortfall.
- [ ] Allow both generator and selector configuration to choose
      `cli: codex|claude_code`; remove the current Claude-only selector
      restriction.

## Embedding provider tasks

- [ ] Define a narrow `EmbeddingProvider` protocol separate from generative
      agent interfaces.
- [ ] Implement `OpenAIEmbeddingProvider` with the official OpenAI SDK and the
      embeddings endpoint only.
- [ ] Do not route ideation embeddings through `LLMBackend.create_embedding`;
      keep the direct API exception visible in the narrow provider type.
- [ ] Resolve `OPENAI_API_KEY` from the process environment after existing
      dotenv startup; never read a machine-specific `.env` path.
- [ ] Default to `text-embedding-3-small` with a configuration override.
- [ ] Never log, persist, or forward the API key to Codex or Claude subprocesses.
- [ ] Embed a canonical descriptor/proposal representation, storing provider,
      model, dimensions, and input hash with each vector.
- [ ] Reuse a vector only when all embedding metadata matches.
- [ ] Compute cosine similarity locally; do not require Weaviate for idea
      duplicate alarms.
- [ ] On missing key, rate limit, timeout, or API error, record
      `embedding_unavailable` and continue exact/descriptor analysis.
- [ ] Record model, call count, input-token usage when returned, duration, and
      failure class as embedding telemetry without hard-coded price claims.

## Analyzer tasks

- [ ] Validate candidate schema, referenced IDs, parent, artifacts, expected
      observation, evaluation method, and resource request.
- [ ] Mark exact duplicates ineligible but retain their records.
- [ ] Compute semantic neighbors through `EmbeddingProvider` and store complete
      compatibility metadata with the result.
- [ ] Compare structured descriptors independently of embedding similarity.
- [ ] Verify evidence references and flag unsupported or contradicted claims.
- [ ] Ask the capacity provider whether implementation plus comparable
      evaluation fits; do not trust coding-agent duration estimates.
- [ ] Summarize operator/descriptor coverage and candidate eligibility.
- [ ] Request at most one repair round when fewer than two distinct eligible
      candidates survive outside `RECOVER`/`VERIFY`.

## Selector tasks

- [ ] Apply deterministic hard rules before constructing the critic prompt.
- [ ] Pass the frozen evidence, analysis facts, and eligible pool to the critic.
- [ ] Require a diagnosis audit for material causal claims.
- [ ] Require selected idea, ordered fallbacks, gap decisions, expected
      benefit/cost, and duplicate overrides.
- [ ] Validate the critic response against eligible IDs.
- [ ] On critic failure, choose only through an explicit deterministic fallback
      policy recorded in the decision; never silently choose the first string.
- [ ] Mark eligible unselected candidates `DEFERRED`; reserve `REJECTED` for an
      explicit negative decision.
- [ ] Never add embedding similarity or self-reported novelty to utility.

## Tests

- All members receive the same evidence snapshot ID and distinct operator
  briefs.
- A member without MCP access still receives mandatory campaign evidence.
- Malformed, empty, skeleton, and duplicate responses remain auditable.
- Semantic neighbor does not automatically disqualify a changed test/parent.
- Exact duplicate cannot win without a materially changed condition.
- Contradicted-claim candidate is ineligible unless it proposes a resolving
  test.
- Repair runs once and targets missing descriptor coverage.
- Selector cannot return an unknown or hard-ineligible ID.
- Selector failure produces a persisted deterministic fallback decision.
- Deferred archived idea can compete with new ideas without being regenerated.
- No generator, repair, selector, or extraction test reaches a direct model
  API; fake CLI runners cover every reasoning call.
- OpenAI embedding calls are isolated behind a fake provider in unit tests.
- Missing embedding credentials degrade cleanly without changing eligibility
  except for the absent semantic warning.
- Vectors from different models or dimensions are never compared.

## Definition of done

- Captured agent outputs can be replayed deterministically through analysis and
  selection.
- All considered candidates and reasoning survive process restart.
- No function in this module creates a SearchNode or writes a Git branch.
- Candidate telemetry preserves existing budget accounting behavior.

## Non-goals

- Campaign state selection.
- Parent branch creation.
- Experiment outcome recording.
- Training a learned candidate ranker.
- Non-OpenAI embedding providers in v1.
