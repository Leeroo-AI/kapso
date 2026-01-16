# Principle: Multi_Rollout_Entry

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Entry_Point]], [[domain::Parallel_Computing]], [[domain::Evaluation]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:30 GMT]]
|}

== Overview ==

Entry point pattern for multi-rollout evaluation systems that orchestrate parallel agent runs with result collection and aggregation.

=== Description ===

Multi-Rollout Entry defines how to structure the main script for parallel agent evaluation:

1. **Dataset loading** - Read questions/tasks from JSONL files
2. **Rollout configuration** - Number of parallel runs per question
3. **Parallel execution** - Spawn agents with async/multiprocessing
4. **Result collection** - Gather outputs from all rollouts
5. **Output persistence** - Save results to JSONL for analysis

This pattern is used by WebResummer and WebSailor evaluation scripts.

=== Usage ===

Use Multi-Rollout Entry when:
- Running batch evaluation experiments
- Need multiple independent agent runs
- Computing Pass@K metrics
- Building reproducible evaluation pipelines

== Theoretical Basis ==

Multi-rollout execution pattern:

'''Multi-Rollout Entry Pattern:'''
<syntaxhighlight lang="python">
import asyncio
import json
from pathlib import Path

async def main(
    input_file: str,
    output_dir: str,
    num_rollouts: int = 5,
    num_workers: int = 10
):
    """
    Run multi-rollout evaluation.

    Args:
        input_file: JSONL with questions
        output_dir: Directory for results
        num_rollouts: Runs per question
        num_workers: Parallel workers
    """
    # Load questions
    questions = [json.loads(line) for line in open(input_file)]

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create task queue
    tasks = []
    for q in questions:
        for rollout_id in range(num_rollouts):
            tasks.append((q, rollout_id))

    # Process with worker pool
    semaphore = asyncio.Semaphore(num_workers)

    async def worker(question, rollout_id):
        async with semaphore:
            result = await run_agent(question['query'])
            save_result(output_dir, question['id'], rollout_id, result)

    await asyncio.gather(*[worker(q, r) for q, r in tasks])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rollouts', type=int, default=5)
    args = parser.parse_args()

    asyncio.run(main(args.input, args.output, args.rollouts))
</syntaxhighlight>

Key components:
- **CLI arguments**: Input/output paths, rollout config
- **Async workers**: Controlled parallelism with semaphore
- **Checkpointing**: Resume from partial runs
- **Progress tracking**: Log completion status

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_WebResummer_Main]]

=== Related Principles ===
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Parallel_Evaluation]]
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Pass_At_K_Evaluation]]
