# Principle: Enhanced_Statistics

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::File|evaluate_deepsearch_official.py|evaluation/evaluate_deepsearch_official.py]]
|-
! Domains
| [[domain::Evaluation]], [[domain::Agent_Analysis]], [[domain::Agent_Systems]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

Statistics filtered by correctness - measures resource usage for correctly solved questions versus all questions.

=== Description ===

Enhanced Statistics extend basic behavioral analysis by conditioning metrics on solution correctness. This enables understanding of how resource consumption differs between successful and unsuccessful problem-solving attempts.

The key insight is that agents may exhibit different behaviors when they successfully solve a problem versus when they fail:
- **Successful attempts** may show more focused, efficient tool usage
- **Failed attempts** may show either too few actions (giving up early) or too many (getting stuck in loops)

By comparing statistics for correctly solved questions against overall statistics, researchers can identify:
- Efficiency patterns correlated with success
- Resource overhead of unsuccessful attempts
- Optimal operating parameters for agents

=== Usage ===

Use Enhanced Statistics when:
- Analyzing efficiency of successful versus failed attempts
- Optimizing agent resource allocation
- Understanding computational cost of correctness
- Identifying patterns in successful problem-solving trajectories

These statistics provide actionable insights for agent improvement.

== Theoretical Basis ==

Enhanced Statistics are computed by filtering behavioral metrics to only include items that received a "Correct" judgement from the LLM judge.

'''Computation Process:'''
<syntaxhighlight lang="text">
Input:
  - round_results: Dict with round1/round2/round3 judgement results
  - round_items: Dict with round1/round2/round3 raw inference items

For each round:
  For each result with "Correct" judgement:
    1. Find matching item by question text
    2. Extract assistant messages from trajectory
    3. Count tool calls (assistant message count)
    4. Count tokens in thinking content
    5. Accumulate to totals

Output:
  - avg_tool_calls_per_question_correctly_solved
  - avg_assistant_tokens_per_question_correctly_solved
</syntaxhighlight>

'''Key Metrics:'''
{| class="wikitable"
|-
! Metric !! Description !! Comparison
|-
| avg_tool_calls_per_question_correctly_solved || Mean tool calls for correct answers || vs avg_tool_calls_per_question
|-
| avg_assistant_tokens_per_question_correctly_solved || Mean tokens for correct answers || vs avg_assistant_tokens_per_question
|}

'''Interpretation Example:'''
<syntaxhighlight lang="text">
Overall Statistics:
  - avg_tool_calls_per_question: 12.5
  - avg_assistant_tokens_per_question: 4500

Enhanced Statistics (correct only):
  - avg_tool_calls_per_question_correctly_solved: 10.2
  - avg_assistant_tokens_per_question_correctly_solved: 3800

Interpretation:
  - Successful attempts use ~18% fewer tool calls
  - Successful attempts use ~16% fewer tokens
  - Failed attempts involve more exploration/retry behavior
</syntaxhighlight>

'''Token Counting:'''
<syntaxhighlight lang="python">
# Uses Qwen tokenizer or falls back to tiktoken
tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-7B")
# or
tokenizer = tiktoken.encoding_for_model("gpt-4o")

# Counts tokens in <think>...</think> content
think_content = content.split('<think>')[-1].split('</think>')[0]
tokens = len(tokenizer.encode(think_content))
</syntaxhighlight>

This analysis helps identify the "cost of correctness" and guides optimization of agent configurations.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_Calculate_Enhanced_Statistics]]
