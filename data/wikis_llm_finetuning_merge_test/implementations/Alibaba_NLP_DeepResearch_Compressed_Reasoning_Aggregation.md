# Implementation: Compressed_Reasoning_Aggregation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Multi_Trajectory]], [[domain::Reasoning]], [[domain::Aggregation]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:00 GMT]]
|}

== Overview ==
Multi-trajectory reasoning aggregation module that synthesizes insights from parallel search paths using compressed reasoning and LLM-based convergence.

=== Description ===
The `compressed_reasoning_aggregation.py` module implements the aggregation phase of ParallelMuse's multi-trajectory search. After multiple rollouts explore different reasoning paths, this module:

- Collects compressed reasoning from all trajectories
- Synthesizes insights using `call_converge` function
- Produces a final answer that incorporates evidence from all paths
- Handles conflicting information through weighted consensus

The aggregation uses an LLM to analyze all trajectory outputs and produce a coherent final response.

=== Usage ===
Call after parallel rollouts complete to aggregate results into a single answer. Essential for the ParallelMuse multi-trajectory search system.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba_NLP_DeepResearch]
* '''File:''' [https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/ParallelMuse/compressed_reasoning_aggregation.py WebAgent/ParallelMuse/compressed_reasoning_aggregation.py]
* '''Lines:''' 1-295

=== Signature ===
<syntaxhighlight lang="python">
def call_converge(
    question: str,
    trajectory_results: List[Dict],
    client: OpenAI,
    model: str = "gpt-4o"
) -> str:
    """
    Aggregate multiple trajectory results into final answer.

    Args:
        question: Original user question
        trajectory_results: List of trajectory outputs with reasoning
        client: OpenAI client for LLM calls
        model: Model for aggregation (default gpt-4o)

    Returns:
        Final aggregated answer string
    """
    ...

def compress_trajectory(
    trajectory: List[Dict]
) -> str:
    """
    Compress a trajectory into condensed reasoning.

    Args:
        trajectory: Full trajectory messages

    Returns:
        Compressed reasoning summary
    """
    ...

def aggregate_evidence(
    compressed_reasonings: List[str],
    question: str
) -> Dict:
    """
    Aggregate evidence from compressed reasonings.

    Args:
        compressed_reasonings: List of compressed trajectory summaries
        question: Original question

    Returns:
        Dict with 'evidence', 'conflicts', 'consensus'
    """
    ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from WebAgent.ParallelMuse.compressed_reasoning_aggregation import (
    call_converge,
    compress_trajectory,
    aggregate_evidence
)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| question || str || Yes || Original research question
|-
| trajectory_results || List[Dict] || Yes || Results from parallel rollouts
|-
| client || OpenAI || Yes || OpenAI client for LLM
|-
| model || str || No || Model name (default gpt-4o)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| final_answer || str || Aggregated answer from all trajectories
|-
| compressed_reasoning || str || Condensed reasoning from trajectory
|-
| evidence_dict || Dict || Aggregated evidence with conflicts
|}

== Usage Examples ==

=== Basic Aggregation ===
<syntaxhighlight lang="python">
from openai import OpenAI
from WebAgent.ParallelMuse.compressed_reasoning_aggregation import call_converge

client = OpenAI(api_key="your-key")

# Results from parallel rollouts
trajectory_results = [
    {
        "answer": "Vienna, Austria",
        "reasoning": "Found on ACL website: venue is Vienna",
        "confidence": 0.9
    },
    {
        "answer": "Vienna",
        "reasoning": "Conference page says Vienna, Austria",
        "confidence": 0.85
    },
    {
        "answer": "Vienna, Austria at Messe Wien",
        "reasoning": "Venue details page shows Messe Wien",
        "confidence": 0.95
    }
]

# Aggregate results
final_answer = call_converge(
    question="Where is ACL 2025 being held?",
    trajectory_results=trajectory_results,
    client=client,
    model="gpt-4o"
)
print(final_answer)
# Output: "ACL 2025 is being held in Vienna, Austria at Messe Wien."
</syntaxhighlight>

=== Compress and Aggregate ===
<syntaxhighlight lang="python">
from WebAgent.ParallelMuse.compressed_reasoning_aggregation import (
    compress_trajectory,
    aggregate_evidence
)

# Compress full trajectories
trajectories = [...]  # Full message histories
compressed = [compress_trajectory(t) for t in trajectories]

# Aggregate evidence
evidence = aggregate_evidence(
    compressed_reasonings=compressed,
    question="What is the ACL 2025 deadline?"
)

print(f"Consensus: {evidence['consensus']}")
print(f"Conflicts: {evidence['conflicts']}")
</syntaxhighlight>

== Related Pages ==
* [[implements::Principle:Alibaba_NLP_DeepResearch_Multi_Trajectory_Aggregation]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_API_Keys_Configuration]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_Python_Dependencies]]
