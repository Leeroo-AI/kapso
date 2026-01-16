# Implementation: Functionality_Specified_Partial_Rollout

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Parallel_Computing]], [[domain::Multi_Agent]], [[domain::Web_Research]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:00 GMT]]
|}

== Overview ==
Parallel MCTS-style rollout orchestrator that executes multiple search trajectories with functionality-aware node expansion and reasoning aggregation for deep web research tasks.

=== Description ===
The `functionality_specified_partial_rollout` module implements a Monte Carlo Tree Search (MCTS) inspired parallel rollout system for web research agents. It manages multiple concurrent search trajectories using async operations, where each trajectory represents a different reasoning path through the search space. The system uses a tree-based structure with node expansion, rollout execution, and results aggregation to find optimal answers to complex questions.

Key components include:
- `Node` class for MCTS tree structure with compressed reasoning context
- Async batch execution of multiple trajectories with `agentic_loop`
- Functionality-aware tool selection that expands only relevant search actions
- Compressed reasoning aggregation that synthesizes insights across trajectories

=== Usage ===
Import this module when implementing parallel multi-trajectory search systems that need to explore diverse reasoning paths simultaneously. Use when a single-path agent is insufficient and you need MCTS-style exploration with partial rollouts.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba_NLP_DeepResearch]
* '''File:''' [https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/ParallelMuse/functionality_specified_partial_rollout.py WebAgent/ParallelMuse/functionality_specified_partial_rollout.py]
* '''Lines:''' 1-526

=== Signature ===
<syntaxhighlight lang="python">
class Node:
    """MCTS tree node with compressed reasoning context."""
    def __init__(self, parent=None):
        self.parent = parent
        self.children = []
        self.state = None
        self.visits = 0
        self.value = 0.0
        self.compressed_reasoning = ""

async def agentic_loop(
    messages: List[Dict],
    client: AsyncOpenAI,
    model: str,
    tools: List[Dict],
    max_iterations: int = 30,
    **kwargs
) -> Dict:
    """Execute a single trajectory rollout with tool calling."""
    ...

async def run_parallel_rollouts(
    question: str,
    num_rollouts: int,
    client: AsyncOpenAI,
    model: str,
    tools: List[Dict],
    **kwargs
) -> List[Dict]:
    """Execute multiple parallel rollouts for a question."""
    ...

def aggregate_results(
    rollout_results: List[Dict],
    question: str
) -> str:
    """Aggregate multiple trajectory results into final answer."""
    ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from WebAgent.ParallelMuse.functionality_specified_partial_rollout import (
    Node,
    agentic_loop,
    run_parallel_rollouts,
    aggregate_results
)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| question || str || Yes || The research question to answer
|-
| num_rollouts || int || Yes || Number of parallel trajectories to execute
|-
| client || AsyncOpenAI || Yes || Async OpenAI client for LLM calls
|-
| model || str || Yes || Model name (e.g., "qwen-max")
|-
| tools || List[Dict] || Yes || Tool definitions for function calling
|-
| max_iterations || int || No || Max steps per trajectory (default 30)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| rollout_results || List[Dict] || List of trajectory results with messages and final answers
|-
| aggregated_answer || str || Final synthesized answer from all trajectories
|-
| compressed_reasoning || str || Compressed reasoning context for each node
|}

== Usage Examples ==

=== Basic Parallel Rollout ===
<syntaxhighlight lang="python">
import asyncio
from openai import AsyncOpenAI
from WebAgent.ParallelMuse.functionality_specified_partial_rollout import (
    run_parallel_rollouts,
    aggregate_results
)

async def main():
    # Initialize async client
    client = AsyncOpenAI(
        api_key="your-api-key",
        base_url="https://api.openai.com/v1"
    )

    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    # Run parallel rollouts
    question = "What are the key features of the Qwen-2.5 model?"
    results = await run_parallel_rollouts(
        question=question,
        num_rollouts=5,
        client=client,
        model="qwen-max",
        tools=tools,
        max_iterations=20
    )

    # Aggregate results
    final_answer = aggregate_results(results, question)
    print(final_answer)

asyncio.run(main())
</syntaxhighlight>

=== Using Node Tree Structure ===
<syntaxhighlight lang="python">
from WebAgent.ParallelMuse.functionality_specified_partial_rollout import Node

# Create root node
root = Node(parent=None)
root.state = {"question": "Complex research question"}
root.compressed_reasoning = "Initial context"

# Expand children for different search strategies
search_child = Node(parent=root)
search_child.state = {"action": "search", "query": "strategy 1"}
root.children.append(search_child)

visit_child = Node(parent=root)
visit_child.state = {"action": "visit", "url": "https://example.com"}
root.children.append(visit_child)

# Track visits and value for MCTS selection
search_child.visits += 1
search_child.value = 0.8  # High value from successful search
</syntaxhighlight>

== Related Pages ==
* [[implements::Principle:Alibaba_NLP_DeepResearch_Parallel_Rollout_Orchestration]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_Python_Dependencies]]
