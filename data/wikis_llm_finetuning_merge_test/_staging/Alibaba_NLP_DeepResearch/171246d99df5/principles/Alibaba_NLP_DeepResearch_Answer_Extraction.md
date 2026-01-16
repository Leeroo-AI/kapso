# Principle: Answer_Extraction

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|ReAct: Synergizing Reasoning and Acting in Language Models|https://arxiv.org/abs/2210.03629]]
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Agent_Systems]], [[domain::Output_Parsing]], [[domain::NLP]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:30 GMT]]
|}

== Overview ==

Answer extraction from agent responses using XML-style tags (`<answer>...</answer>`). This pattern provides a reliable, parseable format for agent final outputs.

=== Description ===

Answer extraction is the final step in the ReAct agent workflow. After the agent has gathered sufficient information through search, webpage visits, and code execution, it produces a final answer within XML-style tags.

The DeepResearch implementation uses a consistent format:
- `<think>` tags for final reasoning before answering
- `<answer>` tags to delimit the extractable answer

This structured approach provides several benefits:

1. **Reliable Parsing** - Simple string splitting extracts the answer
2. **Clear Termination** - Presence of `<answer>` signals loop completion
3. **Reasoning Visibility** - `<think>` tags preserve the agent's final reasoning
4. **Error Handling** - Missing tags indicate incomplete or failed responses

The extraction handles multiple scenarios:
- Normal completion with proper tags
- Forced completion when token limit reached
- Format errors when tags are malformed

=== Usage ===

Use Answer Extraction when:
- Detecting agent completion in the ReAct loop
- Parsing final answers for evaluation
- Handling forced termination scenarios

Answer format requirements:
| Component | Format | Required |
|-----------|--------|----------|
| Thinking | `<think>...</think>` | Recommended |
| Answer | `<answer>...</answer>` | Required |

== Theoretical Basis ==

Answer extraction implements a simple but robust parsing pattern:

<math>
\text{HasAnswer}(R) = \text{contains}(R, \texttt{<answer>}) \land \text{contains}(R, \texttt{</answer>})
</math>

<math>
\text{Extract}(R) = R[\text{indexOf}(\texttt{<answer>}) + 8 : \text{indexOf}(\texttt{</answer>})]
</math>

The termination decision is:
<math>
\text{Terminate} = \text{HasAnswer}(R) \lor \text{LimitReached}
</math>

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# Answer Extraction Pattern
def has_answer(response: str) -> bool:
    """Check if response contains a properly formatted answer."""
    return '<answer>' in response and '</answer>' in response

def extract_answer(response: str) -> Optional[str]:
    """Extract answer from response if present."""
    if not has_answer(response):
        return None

    # Simple split-based extraction
    after_open = response.split('<answer>')[1]
    answer = after_open.split('</answer>')[0]
    return answer.strip()

def determine_termination(response: str, context: Dict) -> Tuple[str, str]:
    """Determine termination reason and extract prediction."""

    if has_answer(response):
        return 'answer', extract_answer(response)

    if context['num_llm_calls'] <= 0:
        return 'exceed available llm calls', 'No answer found.'

    if context['token_count'] > context['max_tokens']:
        # Force answer was requested but not provided
        return 'format error: generate an answer as token limit reached', response

    return 'answer not found', 'No answer found.'

# In the ReAct loop
def process_final_response(messages: List[Dict], context: Dict) -> Dict:
    """Process the final response and determine result."""
    final_response = messages[-1]['content']

    termination, prediction = determine_termination(final_response, context)

    return {
        'question': context['question'],
        'answer': context['ground_truth'],
        'messages': messages,
        'prediction': prediction,
        'termination': termination
    }
</syntaxhighlight>

Key extraction principles:
- **Defensive Parsing**: Always handles missing tags gracefully
- **Termination Priority**: Answer detection takes precedence over other loop conditions
- **Context Preservation**: Full message history is retained for analysis
- **Multiple Exit Paths**: Different termination reasons are tracked for debugging

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_Answer_Extraction_Pattern]]

=== Related Principles ===
* [[related_to::Principle:Alibaba_NLP_DeepResearch_ReAct_Loop_Execution]]
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Context_Management]]
