# Implementation: Answer_Extraction_Pattern

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Agent_Systems]], [[domain::Output_Parsing]], [[domain::Pattern_Documentation]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:30 GMT]]
|}

== Overview ==

Pattern documentation for answer extraction from agent responses using XML-style `<answer>` tags. This is a pattern specification rather than a single function.

=== Description ===

The answer extraction pattern is implemented inline within the `_run()` method of `MultiTurnReactAgent`. It handles multiple scenarios for extracting the agent's final answer from responses.

The pattern consists of:
1. **Answer Detection** - Checking for presence of `<answer>` and `</answer>` tags
2. **Content Extraction** - Using string splitting to extract the answer text
3. **Termination Classification** - Determining the reason the loop ended
4. **Result Packaging** - Assembling the final output dictionary

This pattern is used in three locations within `_run()`:
- Normal loop termination (lines 211-216)
- Forced answer after token limit (lines 196-201)
- LLM call limit exceeded handling (lines 217-218)

=== Usage ===

Use the Answer Extraction Pattern when:
- Parsing agent responses for final answers
- Implementing termination detection in agent loops
- Building evaluation pipelines for agent outputs

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba-NLP/DeepResearch]
* '''File:''' inference/react_agent.py
* '''Lines:''' 211-226

=== Interface Specification ===
<syntaxhighlight lang="python">
# Answer Extraction Pattern - Interface Specification
#
# Input Format:
#   response: str - Agent response that may contain answer tags
#
# Expected Response Format:
#   <think>
#   [Agent's final reasoning about the research findings]
#   </think>
#   <answer>
#   [The final answer to the user's question]
#   </answer>
#
# Output:
#   prediction: str - Extracted answer text
#   termination: str - One of:
#       - "answer" - Normal completion
#       - "answer not found" - No answer tags found
#       - "exceed available llm calls" - Hit call limit
#       - "generate an answer as token limit reached" - Forced answer
#       - "format error: generate an answer as token limit reached" - Bad format
#       - "No answer found after 2h30mins" - Time limit

# Pattern Implementation (from lines 211-226):
if '<answer>' in messages[-1]['content']:
    prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
    termination = 'answer'
else:
    prediction = 'No answer found.'
    termination = 'answer not found'
    if num_llm_calls_available == 0:
        termination = 'exceed available llm calls'

result = {
    "question": question,
    "answer": answer,          # Ground truth
    "messages": messages,      # Full conversation
    "prediction": prediction,  # Extracted answer
    "termination": termination # Exit reason
}
return result
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# The pattern is inline in react_agent.py
# For standalone use, implement as:

def extract_answer_from_response(response: str) -> tuple[str, str]:
    """
    Extract answer from agent response.

    Args:
        response: Agent response string

    Returns:
        Tuple of (prediction, termination_reason)
    """
    if '<answer>' in response and '</answer>' in response:
        prediction = response.split('<answer>')[1].split('</answer>')[0]
        return prediction.strip(), 'answer'
    return 'No answer found.', 'answer not found'
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| response || str || Yes || Agent response text potentially containing answer tags
|-
| messages || List[Dict] || Yes || Full conversation history
|-
| num_llm_calls_available || int || Yes || Remaining LLM call budget
|-
| question || str || Yes || Original user question
|-
| answer || str || Yes || Ground truth answer (for evaluation)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| result || Dict || Complete result dictionary
|-
| result["question"] || str || Original question
|-
| result["answer"] || str || Ground truth answer
|-
| result["messages"] || List[Dict] || Full conversation history
|-
| result["prediction"] || str || Extracted or default prediction
|-
| result["termination"] || str || Termination reason code
|}

== Usage Examples ==

=== Basic Answer Extraction ===
<syntaxhighlight lang="python">
def extract_answer(response: str) -> str:
    """Extract answer from a response string."""
    if '<answer>' in response and '</answer>' in response:
        return response.split('<answer>')[1].split('</answer>')[0].strip()
    return None

# Example responses
response_with_answer = """
<think>
Based on my research, I found multiple sources confirming that Paris is the capital of France.
The city has been the capital since 987 CE.
</think>
<answer>
Paris is the capital of France.
</answer>
"""

response_without_answer = """
<think>
I need to search for more information to answer this question accurately.
</think>
<tool_call>
{"name": "search", "arguments": {"query": "capital of France"}}
</tool_call>
"""

print(extract_answer(response_with_answer))  # "Paris is the capital of France."
print(extract_answer(response_without_answer))  # None
</syntaxhighlight>

=== Full Termination Handling ===
<syntaxhighlight lang="python">
def process_termination(
    messages: list,
    num_llm_calls_available: int,
    question: str,
    ground_truth: str
) -> dict:
    """Process termination and build result dictionary."""

    final_response = messages[-1]['content']

    if '<answer>' in final_response and '</answer>' in final_response:
        prediction = final_response.split('<answer>')[1].split('</answer>')[0]
        termination = 'answer'
    else:
        prediction = 'No answer found.'
        if num_llm_calls_available <= 0:
            termination = 'exceed available llm calls'
        else:
            termination = 'answer not found'

    return {
        "question": question,
        "answer": ground_truth,
        "messages": messages,
        "prediction": prediction,
        "termination": termination
    }

# Usage
result = process_termination(
    messages=conversation_history,
    num_llm_calls_available=0,
    question="What is quantum computing?",
    ground_truth="A type of computation using quantum mechanics"
)

print(f"Termination: {result['termination']}")
print(f"Prediction: {result['prediction'][:100]}...")
</syntaxhighlight>

=== Handling Forced Answers ===
<syntaxhighlight lang="python">
# When token limit is reached, agent is forced to answer
FORCE_ANSWER_PROMPT = """You have now reached the maximum context length you can handle.
You should stop making tool calls and, based on all the information above,
think again and provide what you consider the most likely answer in the
following format:<think>your final thinking</think>
<answer>your answer</answer>"""

def handle_token_limit(messages: list, call_server_fn) -> dict:
    """Handle token limit reached scenario."""

    # Inject force answer prompt
    messages[-1]['content'] = FORCE_ANSWER_PROMPT

    # Get forced response
    response = call_server_fn(messages)
    messages.append({"role": "assistant", "content": response})

    # Extract answer
    if '<answer>' in response and '</answer>' in response:
        prediction = response.split('<answer>')[1].split('</answer>')[0]
        termination = 'generate an answer as token limit reached'
    else:
        prediction = response  # Use full response as fallback
        termination = 'format error: generate an answer as token limit reached'

    return {
        "prediction": prediction,
        "termination": termination,
        "messages": messages
    }
</syntaxhighlight>

=== Evaluation Pipeline Integration ===
<syntaxhighlight lang="python">
import json
from pathlib import Path

def evaluate_results(results_file: str) -> dict:
    """Evaluate agent results from a JSONL file."""

    stats = {
        'total': 0,
        'successful': 0,
        'by_termination': {}
    }

    with open(results_file) as f:
        for line in f:
            result = json.loads(line)
            stats['total'] += 1

            termination = result['termination']
            stats['by_termination'][termination] = \
                stats['by_termination'].get(termination, 0) + 1

            if termination == 'answer':
                stats['successful'] += 1

    stats['success_rate'] = stats['successful'] / stats['total'] * 100

    return stats

# Example output:
# {
#   'total': 100,
#   'successful': 85,
#   'by_termination': {
#     'answer': 85,
#     'exceed available llm calls': 10,
#     'generate an answer as token limit reached': 5
#   },
#   'success_rate': 85.0
# }
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Alibaba_NLP_DeepResearch_Answer_Extraction]]

=== Related Implementations ===
* [[related_to::Implementation:Alibaba_NLP_DeepResearch_MultiTurnReactAgent__run]]
* [[related_to::Implementation:Alibaba_NLP_DeepResearch_count_tokens]]
