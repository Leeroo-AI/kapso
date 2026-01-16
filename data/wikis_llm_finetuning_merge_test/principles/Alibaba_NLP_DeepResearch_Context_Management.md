# Principle: Context_Management

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Lost in the Middle: How Language Models Use Long Contexts|https://arxiv.org/abs/2307.03172]]
* [[source::Paper|Scaling Transformer to 1M tokens with RoPE|https://arxiv.org/abs/2104.09864]]
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Context_Length]], [[domain::Agent_Systems]], [[domain::Token_Management]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:30 GMT]]
|}

== Overview ==

Token counting and context length management for long-running agent sessions. Prevents context overflow by monitoring token usage and triggering graceful termination.

=== Description ===

Context management is critical for autonomous agents that accumulate information over many interaction rounds. As the agent searches, visits webpages, and executes tools, the conversation history grows. Without proper management, the context can exceed the model's maximum length, causing failures.

The DeepResearch implementation includes:

1. **Token Counting** - Uses HuggingFace AutoTokenizer to count tokens in the conversation
2. **Threshold Monitoring** - Checks against a 110K token limit after each round
3. **Graceful Degradation** - When limit is approached, forces the agent to produce a final answer
4. **Chat Template Application** - Applies the model's chat template before counting for accuracy

The token counting mechanism ensures the agent can always produce an answer, even if research is incomplete, rather than failing silently.

=== Usage ===

Use Context Management when:
- Running long agent sessions with many tool calls
- Processing queries that require extensive research
- Preventing context overflow errors

Context limits and actions:
| Threshold | Action |
|-----------|--------|
| < 110K tokens | Continue normal operation |
| >= 110K tokens | Force final answer generation |
| Time > 150 min | Return timeout message |
| LLM calls > 100 | Append limit message |

== Theoretical Basis ==

Context management balances information retention with model limitations:

<math>
\text{TokenCount}(M) = |\text{Tokenize}(\text{ChatTemplate}(M))|
</math>

Where M is the message list and the decision function is:

<math>
\text{Action} = \begin{cases} \text{Continue} & \text{if } \text{TokenCount} < T_{max} \\ \text{ForceAnswer} & \text{if } \text{TokenCount} \geq T_{max} \end{cases}
</math>

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# Context Management Pattern
class ContextManager:
    def __init__(self, model_path: str, max_tokens: int = 110 * 1024):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_tokens = max_tokens

    def count_tokens(self, messages: List[Dict]) -> int:
        """Count tokens in a message list using the model's chat template."""
        # Apply chat template to get the full prompt
        full_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False
        )

        # Tokenize and count
        tokens = self.tokenizer(full_prompt, return_tensors="pt")
        token_count = len(tokens["input_ids"][0])

        return token_count

    def should_force_answer(self, messages: List[Dict]) -> bool:
        """Check if we should force the agent to produce an answer."""
        token_count = self.count_tokens(messages)
        return token_count > self.max_tokens

    def get_force_answer_prompt(self) -> str:
        """Return the prompt that forces answer generation."""
        return (
            "You have now reached the maximum context length you can handle. "
            "You should stop making tool calls and, based on all the information above, "
            "think again and provide what you consider the most likely answer in the "
            "following format:<think>your final thinking</think>\n"
            "<answer>your answer</answer>"
        )
</syntaxhighlight>

Key context management principles:
- **Accurate Counting**: Uses the model's actual tokenizer and chat template
- **Proactive Monitoring**: Checks after each round, not reactively on failure
- **Graceful Degradation**: Always produces some answer rather than failing
- **Transparent Limits**: Logs token counts for debugging and optimization

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_count_tokens]]

=== Related Principles ===
* [[related_to::Principle:Alibaba_NLP_DeepResearch_ReAct_Loop_Execution]]
* [[related_to::Principle:Alibaba_NLP_DeepResearch_Answer_Extraction]]
