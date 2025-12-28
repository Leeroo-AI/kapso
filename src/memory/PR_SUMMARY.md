# Pull Request: Cognitive Memory Cleanup (post-merge)

## Overview

This PR is a **cleanup pass** on the Cognitive Memory system after the initial implementation landed in `main`.

The goal is to:
- remove legacy/orphan code and unused prompt templates,
- ensure the end-to-end logic is consistent (Tier 1/2/3 → context → decisions → Expert/orchestrator),
- enforce **fail-fast** behavior where we previously had misleading “fallback” scaffolding,
- keep logs + tests high-signal for validating retrieval quality and knowledge passing.

## Key Changes

### 0) End-to-end correctness and simplification

- **Single knowledge representation**: the cognitive loop uses `KGKnowledge` as the only structure for Tier 1/2/3 retrieval and rendering.
- **No workflow state machine**: removed legacy `WorkflowState`/`StepState` style tracking. The agent consumes full knowledge as text; step-state was redundant and confusing.
- **No hardcoded truncations**: removed hardcoded `[:N]` truncations in the knowledge/context path. Any length limits are config-driven where required (e.g. error tail length for query generation).
- **Fail-fast retrieval**: removed silent fallbacks where they masked missing components (LLM unavailable / semantic search returning nothing).

### 1) Remove unused prompt templates

`src/memory/prompts/` now contains **only prompt files that are actually loaded by runtime code**:
- `decide_action.md` (DecisionMaker)
- `extract_error_insight.md` (InsightExtractor)
- `extract_success_insight.md` (InsightExtractor)

Removed unused `.md` prompts that were remnants of an older workflow/state-machine design:
- `infer_failing_step.md`, `should_consult_kg.md`, `workflow_action.md`, `error_response.md`
- `synthesize_plan.md` (old Tier 2 “synthesized workflow” concept)
- `extract_insight.md` (superseded by `extract_error_insight.md` / `extract_success_insight.md`)
- `episodic_retrieval_query.md` (current EpisodicRetriever uses inline prompts)

### 2) Remove unused report generator

Deleted `src/memory/report_generator.py` because it had **no runtime call sites/importers** and was not part of the current execution path.

### 3) Remove dead / contradictory Tier 3 fallback code

Tier 3 is intended to be:
**LLM query generation → KG semantic search → add error knowledge**, with no silent keyword/Neo4j fallbacks.

This PR removes an unused Tier 3 Neo4j fallback helper from `src/memory/knowledge_retriever.py` and keeps Tier 3 fail-fast.

### 4) Ensure KGKnowledge types are tracked

Adds `src/memory/kg_types.py` as a real tracked module (previously present in the working tree but not committed).
This file is the single source of truth for `KGKnowledge` and tier rendering.

### 5) Retrieval quality + plumbing fixes

- **Tier 1/2 query generation**: Tier 1 and Tier 2 use LLM-generated search queries (no hardcoded examples in prompts) before graph traversal.
- **Tier 3 search quality**: Tier 3 includes `Environment` pages in semantic search so ImportError/missing dependency cases retrieve install/setup guidance.
- **Context passing**: cognitive context manager passes the full rendered `KGKnowledge` into agent context (`additional_info`) and avoids duplicating/truncating code snippets.
- **Expert/orchestrator wiring**: ensure the configured KG backend instance is actually injected/used end-to-end (no accidental backend mismatch).

## Testing

Run tests using the correct conda environment and `.env`:

```bash
conda run -n praxium_conda bash -lc 'cd /home/ubuntu/praxium && set -a && source .env && set +a && PYTHONPATH=. python -m compileall -q src tests'
conda run -n praxium_conda bash -lc 'cd /home/ubuntu/praxium && set -a && source .env && set +a && PYTHONPATH=. python tests/test_cognitive_iteration_loop.py'
conda run -n praxium_conda bash -lc 'cd /home/ubuntu/praxium && set -a && source .env && set +a && PYTHONPATH=. python tests/test_expert_full_e2e.py'
```


