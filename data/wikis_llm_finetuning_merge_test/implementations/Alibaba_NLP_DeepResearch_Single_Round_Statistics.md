# Implementation: Single_Round_Statistics

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

Concrete tool for computing behavioral statistics from a single round of agent inference results.

=== Description ===

The `single_round_statistics()` function analyzes agent trajectories to compute comprehensive behavioral metrics. It processes the complete message history from each inference run to extract tool usage patterns, token consumption, and termination reasons.

Key analysis dimensions:
- **Tool usage**: Counts of search, visit, and other tool invocations
- **Response metrics**: Answer length, thinking/reasoning length
- **Token consumption**: Assistant tokens per question and per message
- **Termination analysis**: Frequency of different ending conditions
- **Quality indicators**: Invalid responses, context limit exceedances

The function uses the Qwen tokenizer when available, falling back to tiktoken (GPT-4o encoding) for token counting.

=== Usage ===

Use `single_round_statistics()` when:
- Analyzing agent behavior for a single evaluation round
- Computing input for aggregate statistics across rounds
- Diagnosing agent performance issues
- Understanding resource consumption patterns

This function is called once per round, with results averaged across rounds by `aggregate_statistics()`.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba-NLP/DeepResearch]
* '''File:''' evaluation/evaluate_deepsearch_official.py
* '''Lines:''' 209-325

=== Signature ===
<syntaxhighlight lang="python">
def single_round_statistics(input_file: str) -> Dict:
    """
    Compute behavioral statistics for one evaluation round.

    Args:
        input_file: str - Path to JSONL inference results file

    Returns:
        Dict containing:
            - extra_length: int - Questions exceeding 30k tokens
            - num_invalid: int - Responses without valid <answer> tags
            - avg_action: float - Mean tool calls per question
            - avg_visit_action: float - Mean visit tool calls
            - avg_search_action: float - Mean search tool calls
            - avg_other_action: float - Mean other tool calls
            - avg_ans_length: float - Mean answer character length
            - avg_think_length: float - Mean thinking character length
            - avg_tool_calls_per_question: float - Same as avg_action
            - avg_assistant_tokens_per_question: float - Mean LLM output tokens
            - avg_assistant_tokens_per_message: float - Mean tokens per turn
            - termination_freq: Dict[str, float] - Termination reason frequencies
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from evaluate_deepsearch_official import single_round_statistics
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| input_file || str || Yes || Path to JSONL file with inference results
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| extra_length || int || Count of questions exceeding 30k context tokens
|-
| num_invalid || int || Count of responses missing valid <answer> tags
|-
| avg_action || float || Mean total tool calls per question
|-
| avg_visit_action || float || Mean visit/browse tool calls per question
|-
| avg_search_action || float || Mean search tool calls per question
|-
| avg_other_action || float || Mean other tool calls per question
|-
| avg_ans_length || float || Mean answer character length
|-
| avg_think_length || float || Mean reasoning character length per message
|-
| avg_tool_calls_per_question || float || Same as avg_action
|-
| avg_assistant_tokens_per_question || float || Mean total assistant tokens per question
|-
| avg_assistant_tokens_per_message || float || Mean assistant tokens per individual message
|-
| termination_freq || Dict[str, float] || Distribution of termination reasons
|}

== Usage Examples ==

=== Basic Statistics Computation ===
<syntaxhighlight lang="python">
from evaluate_deepsearch_official import single_round_statistics

# Compute statistics for one round
stats = single_round_statistics("results/iter1.jsonl")

print(f"Average tool calls: {stats['avg_action']:.2f}")
print(f"  - Search: {stats['avg_search_action']:.2f}")
print(f"  - Visit: {stats['avg_visit_action']:.2f}")
print(f"  - Other: {stats['avg_other_action']:.2f}")
print(f"Average answer length: {stats['avg_ans_length']:.2f} chars")
print(f"Invalid responses: {stats['num_invalid']}")
</syntaxhighlight>

=== Aggregating Across All Rounds ===
<syntaxhighlight lang="python">
from evaluate_deepsearch_official import single_round_statistics

# Compute for each round
round1_stats = single_round_statistics("results/iter1.jsonl")
round2_stats = single_round_statistics("results/iter2.jsonl")
round3_stats = single_round_statistics("results/iter3.jsonl")

# Average across rounds
avg_tool_calls = (
    round1_stats['avg_action'] +
    round2_stats['avg_action'] +
    round3_stats['avg_action']
) / 3

print(f"Average tool calls (all rounds): {avg_tool_calls:.2f}")
</syntaxhighlight>

=== Analyzing Termination Patterns ===
<syntaxhighlight lang="python">
stats = single_round_statistics("results/iter1.jsonl")

print("Termination Frequencies:")
for reason, freq in stats['termination_freq'].items():
    print(f"  {reason}: {freq:.1%}")

# Example output:
# Termination Frequencies:
#   answered: 85.0%
#   max_turns_reached: 10.0%
#   max_tokens_reached: 3.0%
#   unknown: 2.0%
</syntaxhighlight>

=== Token Consumption Analysis ===
<syntaxhighlight lang="python">
stats = single_round_statistics("results/iter1.jsonl")

print(f"Token Consumption:")
print(f"  Per question: {stats['avg_assistant_tokens_per_question']:.0f} tokens")
print(f"  Per message: {stats['avg_assistant_tokens_per_message']:.0f} tokens")
print(f"  Context overflows: {stats['extra_length']} questions")

# Estimate cost (assuming GPT-4o pricing)
tokens_per_question = stats['avg_assistant_tokens_per_question']
num_questions = 100  # example
total_tokens = tokens_per_question * num_questions
cost = total_tokens / 1000 * 0.015  # $0.015 per 1K output tokens
print(f"  Estimated output cost: ${cost:.2f}")
</syntaxhighlight>

=== Quality Diagnostics ===
<syntaxhighlight lang="python">
stats = single_round_statistics("results/iter1.jsonl")

# Check for issues
issues = []
if stats['num_invalid'] > 0:
    issues.append(f"{stats['num_invalid']} invalid responses (no <answer> tag)")
if stats['extra_length'] > 0:
    issues.append(f"{stats['extra_length']} context overflows")
if stats['termination_freq'].get('max_turns_reached', 0) > 0.1:
    issues.append("High rate of max_turns termination")

if issues:
    print("Potential Issues:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("No significant issues detected")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Alibaba_NLP_DeepResearch_Behavioral_Statistics]]
