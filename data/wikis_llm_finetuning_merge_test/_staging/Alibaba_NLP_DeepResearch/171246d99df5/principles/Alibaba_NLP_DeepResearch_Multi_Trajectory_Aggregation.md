# Principle: Multi_Trajectory_Aggregation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Self-Consistency|https://arxiv.org/abs/2203.11171]]
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Multi_Agent]], [[domain::Result_Aggregation]], [[domain::Reasoning]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:30 GMT]]
|}

== Overview ==

Compressed reasoning aggregation that synthesizes insights from multiple parallel agent trajectories into a coherent final answer.

=== Description ===

Multi-Trajectory Aggregation addresses the challenge of combining results from parallel rollouts. When multiple agents explore different paths:

1. **Collect results** - Gather final answers from all trajectories
2. **Extract reasoning** - Parse compressed reasoning from each path
3. **Identify consensus** - Find common answers/facts across trajectories
4. **Resolve conflicts** - Handle contradictory information
5. **Synthesize** - Produce unified answer with supporting evidence

The ParallelMuse implementation uses LLM-based aggregation with compressed reasoning context.

=== Usage ===

Use Multi-Trajectory Aggregation when:
- Running parallel agent rollouts
- Need to combine diverse search results
- Want to increase answer reliability
- Building ensemble agent systems

== Theoretical Basis ==

Aggregation can use voting or synthesis:

<math>
\text{Answer} = \text{Synthesize}(\{(a_i, r_i)\}_{i=1}^N)
</math>

Where a_i is answer from trajectory i and r_i is its reasoning.

'''Aggregation Pattern:'''
<syntaxhighlight lang="python">
async def call_converge(
    question: str,
    rollout_results: list[dict],
    client: AsyncOpenAI,
    model: str
) -> str:
    """
    Aggregate multiple trajectory results.

    Args:
        question: Original question
        rollout_results: List of trajectory outputs
        client: OpenAI client
        model: Model name

    Returns:
        Synthesized final answer
    """
    # Format results for LLM
    formatted = "\n\n".join([
        f"Trajectory {i+1}:\n"
        f"Answer: {r['answer']}\n"
        f"Reasoning: {r['compressed_reasoning']}"
        for i, r in enumerate(rollout_results)
    ])

    # Ask LLM to synthesize
    prompt = f"""Given multiple search trajectories for the question:
{question}

Trajectories:
{formatted}

Synthesize the most accurate answer considering all evidence."""

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
</syntaxhighlight>

Aggregation strategies:
- **Majority voting**: Most common answer wins
- **Weighted voting**: Weight by trajectory confidence
- **LLM synthesis**: LLM combines all evidence
- **Evidence pooling**: Merge unique facts, deduplicate

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_Compressed_Reasoning_Aggregation]]

=== Related Principles ===
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Parallel_Rollout_Orchestration]]
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Answer_Extraction]]
