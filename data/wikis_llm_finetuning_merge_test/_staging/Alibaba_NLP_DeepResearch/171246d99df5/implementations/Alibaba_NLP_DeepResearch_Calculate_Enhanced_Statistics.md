# Implementation: Calculate_Enhanced_Statistics

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

Concrete tool for computing behavioral statistics filtered to correctly solved questions only.

=== Description ===

The `calculate_enhanced_statistics()` function computes resource consumption metrics specifically for questions that the agent answered correctly. By comparing these statistics to overall statistics, researchers can understand the efficiency difference between successful and failed problem-solving attempts.

The function:
1. Iterates through all three evaluation rounds
2. Filters to only "Correct" judgements
3. Matches each correct result to its original inference item
4. Extracts tool calls and token counts from the trajectory
5. Computes averages across all correctly solved instances

This enables analysis questions like:
- Do successful attempts use fewer/more tool calls?
- Is there a token budget associated with correct answers?
- How much overhead comes from failed attempts?

=== Usage ===

Use `calculate_enhanced_statistics()` when:
- Analyzing efficiency of successful problem-solving
- Comparing resource usage between correct and incorrect attempts
- Optimizing agent configurations for cost-effectiveness
- Understanding computational cost of correctness

This function provides insights beyond basic accuracy metrics.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba-NLP/DeepResearch]
* '''File:''' evaluation/evaluate_deepsearch_official.py
* '''Lines:''' 328-379

=== Signature ===
<syntaxhighlight lang="python">
def calculate_enhanced_statistics(
    round_results: Dict[str, List[Dict]],
    round_items: Dict[str, List[Dict]]
) -> Dict:
    """
    Compute statistics filtered by correctness.

    Args:
        round_results: Dict with keys "round1", "round2", "round3"
            Each value is a list of judgement dicts with "question" and "judgement" fields
        round_items: Dict with keys "round1", "round2", "round3"
            Each value is a list of original inference items with "messages" field

    Returns:
        Dict containing:
            - avg_tool_calls_per_question_correctly_solved: float
            - avg_assistant_tokens_per_question_correctly_solved: float
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from evaluate_deepsearch_official import calculate_enhanced_statistics
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| round_results || Dict[str, List[Dict]] || Yes || Judgement results per round
|-
| round_results["round1"] || List[Dict] || Yes || Round 1 judgement results
|-
| round_results["round2"] || List[Dict] || Yes || Round 2 judgement results
|-
| round_results["round3"] || List[Dict] || Yes || Round 3 judgement results
|-
| round_items || Dict[str, List[Dict]] || Yes || Original inference items per round
|-
| round_items["round1"] || List[Dict] || Yes || Round 1 inference items with messages
|-
| round_items["round2"] || List[Dict] || Yes || Round 2 inference items with messages
|-
| round_items["round3"] || List[Dict] || Yes || Round 3 inference items with messages
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| avg_tool_calls_per_question_correctly_solved || float || Mean assistant turns for correct answers
|-
| avg_assistant_tokens_per_question_correctly_solved || float || Mean thinking tokens for correct answers
|}

== Usage Examples ==

=== Basic Enhanced Statistics ===
<syntaxhighlight lang="python">
from evaluate_deepsearch_official import (
    calculate_enhanced_statistics,
    process_single_round
)

# Load inference items
round_items = {
    "round1": process_single_round("results/iter1.jsonl"),
    "round2": process_single_round("results/iter2.jsonl"),
    "round3": process_single_round("results/iter3.jsonl")
}

# round_results populated after LLM judge evaluation
round_results = {
    "round1": [...],  # List of {question, answer, judgement}
    "round2": [...],
    "round3": [...]
}

# Calculate enhanced statistics
enhanced = calculate_enhanced_statistics(round_results, round_items)

print(f"Avg tool calls (correct only): {enhanced['avg_tool_calls_per_question_correctly_solved']:.2f}")
print(f"Avg tokens (correct only): {enhanced['avg_assistant_tokens_per_question_correctly_solved']:.0f}")
</syntaxhighlight>

=== Comparing Overall vs Correct-Only Statistics ===
<syntaxhighlight lang="python">
from evaluate_deepsearch_official import (
    aggregate_statistics,
    calculate_enhanced_statistics
)

# Compute overall statistics
overall_stats = aggregate_statistics(
    "results/iter1.jsonl",
    "results/iter2.jsonl",
    "results/iter3.jsonl"
)

# Compute enhanced statistics
enhanced_stats = calculate_enhanced_statistics(round_results, round_items)

# Compare
print("=== Resource Usage Comparison ===")
print(f"Tool calls (all questions):     {overall_stats['avg_tool_calls_per_question']:.2f}")
print(f"Tool calls (correct only):      {enhanced_stats['avg_tool_calls_per_question_correctly_solved']:.2f}")
print(f"Tokens (all questions):         {overall_stats['avg_assistant_tokens_per_question']:.0f}")
print(f"Tokens (correct only):          {enhanced_stats['avg_assistant_tokens_per_question_correctly_solved']:.0f}")

# Calculate efficiency metrics
tool_efficiency = (
    1 - enhanced_stats['avg_tool_calls_per_question_correctly_solved'] /
    overall_stats['avg_tool_calls_per_question']
) * 100
print(f"\nCorrect attempts use {tool_efficiency:.1f}% fewer tool calls")
</syntaxhighlight>

=== Full Evaluation Report ===
<syntaxhighlight lang="python">
# Complete evaluation pipeline output
print(f"=== EVALUATION RESULTS ===")
print(f"Pass@3: {pass_at_3}%")
print(f"Best Pass@1: {best_pass_at_1}%")
print(f"Avg Pass@3: {avg_pass_at_3}%")

print(f"\n=== BEHAVIORAL STATISTICS ===")
print(f"Avg. Tool Calls per Question: {overall_stats['avg_tool_calls_per_question']:.2f}")
print(f"Avg. Tool Calls per Question (Correctly Solved): {enhanced_stats['avg_tool_calls_per_question_correctly_solved']:.2f}")
print(f"Avg. Assistant Tokens per Question: {overall_stats['avg_assistant_tokens_per_question']:.2f}")
print(f"Avg. Assistant Tokens per Question (Correctly Solved): {enhanced_stats['avg_assistant_tokens_per_question_correctly_solved']:.2f}")
</syntaxhighlight>

=== Understanding the Calculation ===
<syntaxhighlight lang="python">
# The function counts assistant messages as "tool calls"
# and tokens from <think>...</think> sections

# For each correct result:
for result in round_results[round_name]:
    if is_correct_judgement(result["judgement"]):
        # Find matching inference item
        item = find_by_question(round_items[round_name], result["question"])

        # Count assistant messages as tool calls
        num_tool_use = sum(1 for m in item["messages"] if m["role"] == "assistant")

        # Count tokens in thinking sections
        tokens = 0
        for msg in item["messages"]:
            if msg["role"] == "assistant":
                think_content = msg["content"].split('<think>')[-1].split('</think>')[0]
                tokens += tokenizer.encode(think_content)

        correct_tool_calls.append(num_tool_use)
        correct_tokens.append(tokens)

# Return averages
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Alibaba_NLP_DeepResearch_Enhanced_Statistics]]
