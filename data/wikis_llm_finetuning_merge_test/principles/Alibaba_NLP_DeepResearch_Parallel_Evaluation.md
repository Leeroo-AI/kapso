# Principle: Parallel_Evaluation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Evaluation]], [[domain::Parallel_Computing]], [[domain::Batch_Processing]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:30 GMT]]
|}

== Overview ==

Parallel batch evaluation runner that executes multiple agent rollouts across a question dataset with controlled concurrency and result aggregation.

=== Description ===

Parallel Evaluation orchestrates large-scale agent evaluation:

1. **Dataset iteration** - Process questions from evaluation set
2. **Worker pool** - Controlled parallelism with semaphores
3. **Progress tracking** - Monitor completion across questions
4. **Result collection** - Aggregate outputs per question
5. **Checkpointing** - Resume from partial runs

The WebSailor implementation uses async workers for efficient evaluation.

=== Usage ===

Use Parallel Evaluation when:
- Running evaluation on benchmark datasets
- Need to process many questions efficiently
- Computing aggregate metrics (accuracy, Pass@K)
- Building reproducible evaluation pipelines

== Theoretical Basis ==

Parallel evaluation pattern:

'''Parallel Evaluation Pattern:'''
<syntaxhighlight lang="python">
import asyncio
from concurrent.futures import ProcessPoolExecutor

async def parallel_evaluate(
    dataset: list[dict],
    agent_factory: callable,
    num_rollouts: int = 5,
    max_workers: int = 10,
    output_path: str = None
):
    """
    Run parallel evaluation.

    Args:
        dataset: List of evaluation questions
        agent_factory: Function to create agent instance
        num_rollouts: Runs per question
        max_workers: Maximum concurrent workers
        output_path: Path for result JSONL
    """
    semaphore = asyncio.Semaphore(max_workers)
    results = []

    async def evaluate_question(question: dict, rollout_id: int):
        async with semaphore:
            agent = agent_factory()
            try:
                answer = await agent.run(question['query'])
                return {
                    'question_id': question['id'],
                    'rollout_id': rollout_id,
                    'answer': answer,
                    'status': 'success'
                }
            except Exception as e:
                return {
                    'question_id': question['id'],
                    'rollout_id': rollout_id,
                    'error': str(e),
                    'status': 'failed'
                }

    # Create all tasks
    tasks = [
        evaluate_question(q, r)
        for q in dataset
        for r in range(num_rollouts)
    ]

    # Run with progress tracking
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        if output_path:
            append_jsonl(output_path, result)

    return results
</syntaxhighlight>

Key features:
- **Async concurrency**: Efficient I/O-bound parallelism
- **Backpressure**: Semaphore limits active workers
- **Fault tolerance**: Continue on individual failures
- **Incremental output**: Stream results to disk

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_WebSailor_Main]]

=== Related Principles ===
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Multi_Rollout_Entry]]
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Pass_At_K_Evaluation]]
