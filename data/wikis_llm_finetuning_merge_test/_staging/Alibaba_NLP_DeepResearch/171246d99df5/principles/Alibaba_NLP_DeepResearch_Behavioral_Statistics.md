# Principle: Behavioral_Statistics

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

Agent behavior analysis including tool usage patterns, token consumption, and termination reasons.

=== Description ===

Behavioral Statistics provide quantitative insights into how research agents operate beyond simple correctness metrics. This analysis examines the computational and interaction patterns that characterize agent behavior during inference.

Key behavioral dimensions tracked:

1. **Tool Usage Patterns**
   - Total actions per question
   - Search tool invocations
   - Visit/browse tool invocations
   - Other tool types

2. **Response Characteristics**
   - Average answer length
   - Average thinking/reasoning length
   - Tokens per assistant message

3. **Resource Consumption**
   - Tool calls per question
   - Total assistant tokens per question
   - Questions exceeding context limits

4. **Termination Analysis**
   - Answered (successful completion)
   - Max turns reached (iteration limit)
   - Max tokens reached (context limit)
   - Unknown termination

=== Usage ===

Use Behavioral Statistics when:
- Diagnosing agent efficiency and resource usage
- Comparing agent architectures or configurations
- Identifying bottlenecks in agent workflows
- Understanding failure modes (why agents don't answer)

These statistics complement accuracy metrics to provide a complete picture of agent performance.

== Theoretical Basis ==

Behavioral Statistics are computed by analyzing the complete message trajectories from agent inference runs.

'''Computation Approach:'''
<syntaxhighlight lang="text">
For each inference item:
  1. Parse messages array from trajectory
  2. Extract tool calls from <tool_call> tags
  3. Classify tools by type (search, visit, other)
  4. Count tokens using tokenizer
  5. Determine termination reason
  6. Aggregate into per-question statistics
</syntaxhighlight>

'''Tool Call Extraction:'''
<syntaxhighlight lang="python">
# Tool calls are embedded in assistant messages
for msg in messages:
    if msg['role'] == 'assistant':
        # Parse <tool_call>{"name": "search", ...}</tool_call>
        # Classify by tool name
</syntaxhighlight>

'''Statistics Computed:'''
{| class="wikitable"
|-
! Statistic !! Description !! Typical Range
|-
| avg_action || Mean tool calls per question || 5-20
|-
| avg_visit_action || Mean page visits per question || 3-15
|-
| avg_search_action || Mean searches per question || 2-5
|-
| avg_other_action || Mean other tools per question || 0-2
|-
| avg_ans_length || Mean answer character length || 50-500
|-
| avg_think_length || Mean reasoning character length || 500-5000
|-
| avg_tool_calls_per_question || Same as avg_action || 5-20
|-
| avg_assistant_tokens_per_question || Total LLM output tokens || 1000-10000
|-
| termination_freq || Distribution of end reasons || varies
|}

'''Termination Detection:'''
<syntaxhighlight lang="text">
1. Check explicit "termination" field if present
2. Else analyze final message content:
   - Contains <answer>...</answer> -> "answered"
   - Contains "max_turns_reached" -> "max_turns_reached"
   - Contains "max_tokens_reached" -> "max_tokens_reached"
   - Otherwise -> "unknown"
</syntaxhighlight>

These statistics are aggregated across all items and averaged across the 3 evaluation rounds.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_Single_Round_Statistics]]
