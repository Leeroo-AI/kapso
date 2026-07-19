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
  coding_agents.py
  generator.py
  embeddings.py
  analyzer.py
  selector.py

generic/prompts/
  ideation_v3_candidate.md
  ideation_v3_selector.md

tests/
  test_ideation_coding_agents.py
  test_ideation_embeddings.py
  test_ideation_generator.py
  test_ideation_analyzer.py
  test_ideation_selector.py
  test_ideation_boundaries.py
```

## Generator tasks

- [x] Define one `CodingAgentCallRunner` contract implemented by Codex CLI and
      Claude Code runners.
- [x] Prohibit direct generative calls through `LLMBackend`, LiteLLM, or an
      OpenAI/Anthropic model API inside the candidate pipeline.
- [x] Enforce read-only CLI operation and preserve timeout, token, cost, and
      complete invocation artifacts. Failed calls propagate without salvage.
- [x] Push the mandatory evidence packet into every member prompt.
- [x] Assign one operator brief and frozen parent snapshot per member.
- [x] Keep members independent until all proposals are collected.
- [x] Parse structured candidates without treating arbitrary full output as a
      valid proposal.
- [x] Persist every structurally valid returned candidate before selection;
      semantic invalidity remains a recorded analysis result.
- [x] Preserve every completed call's artifacts, but propagate any member
      failure and leave the batch `PLANNED` instead of persisting a partial pool.
- [x] Allow both generator and selector configuration to choose
      `cli: codex|claude_code`; remove the current Claude-only selector
      restriction.

## Embedding provider tasks

- [x] Define a narrow `EmbeddingProvider` protocol separate from generative
      agent interfaces.
- [x] Implement `OpenAIEmbeddingProvider` with the official OpenAI SDK and the
      embeddings endpoint only.
- [x] Do not route ideation embeddings through `LLMBackend.create_embedding`;
      keep the direct API exception visible in the narrow provider type.
- [x] Let outer startup and the official SDK discover credentials; ideation
      code never reads an environment variable or `.env` path.
- [x] Default to `text-embedding-3-small` with a configuration override.
- [x] Never log, persist, or forward the API key to Codex or Claude subprocesses.
- [x] Embed a canonical descriptor/proposal representation, storing provider,
      model, dimensions, and input hash with each vector.
- [x] Reuse a vector only when all embedding metadata matches.
- [x] Compute cosine similarity locally; do not require Weaviate for idea
      duplicate alarms.
- [x] Propagate missing credentials, rate limits, timeouts, and API errors;
      skip semantic analysis only when embeddings are explicitly disabled.
- [x] Record model, call count, input-token usage when returned, and duration
      without hard-coded price claims.

## Analyzer tasks

- [x] Validate candidate schema, referenced IDs, parent, artifacts, expected
      observation, evaluation method, and resource request.
- [x] Mark exact duplicates ineligible unless code derives a changed parent,
      evaluation, or evidence condition; retain their records.
- [x] Compute semantic neighbors through `EmbeddingProvider` and store complete
      compatibility metadata with the result.
- [x] Compare structured descriptors independently of embedding similarity.
- [x] Verify evidence references and flag unsupported or contradicted claims.
- [x] Ask the capacity provider whether implementation plus comparable
      evaluation fits; do not trust coding-agent duration estimates.
- [x] Summarize operator/descriptor coverage and candidate eligibility.
- [x] Request at most one repair round when fewer than two distinct eligible
      candidates survive outside `RECOVER`/`VERIFY`.

## Selector tasks

- [x] Apply deterministic hard rules before constructing the critic prompt.
- [x] Pass the frozen evidence, analysis facts, and eligible pool to the critic.
- [x] Require a diagnosis audit for material causal claims.
- [x] Require selected idea, ordered fallbacks, gap decisions, expected
      benefit/cost, and duplicate overrides.
- [x] Validate the critic response against eligible IDs.
- [x] Propagate critic failure and leave the batch `ANALYZED`; never invent a
      fallback winner.
- [x] Mark eligible unselected candidates `DEFERRED`; reserve `REJECTED` for an
      explicit negative decision.
- [x] Never add embedding similarity or self-reported novelty to utility.

## Tests

- All members receive the same evidence snapshot ID and distinct operator
  briefs.
- A member without MCP access still receives mandatory campaign evidence.
- Failed, empty, and malformed calls retain invocation artifacts and propagate;
  structured but semantically invalid candidates remain auditable.
- Semantic neighbor does not automatically disqualify a changed test/parent.
- Exact duplicate cannot win without a materially changed condition.
- Contradicted-claim candidate is ineligible unless it proposes a resolving
  test.
- Repair runs once and targets missing descriptor coverage.
- Selector cannot return an unknown or hard-ineligible ID.
- Selector failure leaves the analyzed batch without a decision.
- Deferred archived idea can compete with new ideas without being regenerated.
- No generator, repair, selector, or extraction test reaches a direct model
  API; fake CLI runners cover every reasoning call.
- OpenAI embedding calls are isolated behind a fake provider in unit tests.
- Missing embedding credentials and provider failures propagate; explicit
  `enabled: false` removes semantic warnings without changing hard eligibility.
- Vectors from different models or dimensions are never compared.

## Definition of done

- Captured agent outputs can be replayed deterministically through analysis and
  selection.
- All considered candidates and reasoning survive process restart.
- No function in this module creates a SearchNode or writes a Git branch.
- Candidate telemetry is complete and exposed for M4 budget accounting.

## Non-goals

- Campaign state selection.
- Parent branch creation.
- Experiment outcome recording.
- Training a learned candidate ranker.
- Non-OpenAI embedding providers in v1.
