{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
* [[source::Doc|OpenAI API Reference|https://platform.openai.com/docs/api-reference/completions]]
|-
! Domains
| [[domain::NLP]], [[domain::Output_Processing]], [[domain::Data_Structures]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

The practice of extracting, interpreting, and utilizing generated text, token sequences, and associated metadata from LLM inference results.

=== Description ===

Output Processing encompasses the steps after text generation to extract useful information from model outputs. This includes:

1. **Text Extraction:** Retrieving the generated string content
2. **Token Analysis:** Examining individual tokens and their IDs
3. **Probability Inspection:** Analyzing log probabilities for confidence assessment
4. **Finish Reason Handling:** Understanding why generation stopped
5. **Multi-Completion Aggregation:** Managing multiple outputs per request
6. **Metadata Utilization:** Using metrics for debugging and optimization

Proper output processing is essential for building robust applications that can handle edge cases, assess quality, and provide debugging information.

=== Usage ===

Apply output processing techniques when:
- Extracting generated text for downstream tasks
- Assessing model confidence via log probabilities
- Handling multiple completion variants
- Debugging generation quality issues
- Building quality control pipelines
- Measuring inference performance

== Theoretical Basis ==

'''Output Structure:'''

vLLM's output follows a hierarchical structure:

<syntaxhighlight lang="python">
# Conceptual output hierarchy
RequestOutput
├── request_id: str          # Unique identifier
├── prompt: str              # Original input
├── prompt_token_ids: [int]  # Tokenized prompt
├── outputs: [               # One per n in SamplingParams
│   └── CompletionOutput
│       ├── text: str        # Generated text
│       ├── token_ids: [int] # Generated tokens
│       ├── logprobs: [dict] # Per-token probabilities
│       ├── finish_reason: str
│       └── stop_reason: str|int
│   ]
├── finished: bool
└── metrics: RequestMetrics
</syntaxhighlight>

'''Log Probability Interpretation:'''

Log probabilities indicate model confidence:

<math>
logprob(token) = \log P(token | context)
</math>

Common uses:
- **Confidence scoring:** Higher logprobs = more confident
- **Perplexity calculation:** <math>PPL = \exp(-\frac{1}{N}\sum logprob)</math>
- **Calibration:** Comparing logprob to empirical accuracy

'''Finish Reasons:'''

<syntaxhighlight lang="python">
# Finish reason meanings
def interpret_finish_reason(completion):
    if completion.finish_reason == "stop":
        # Hit a stop string or stop token
        # stop_reason tells you which one
        return f"Stopped by: {completion.stop_reason}"

    elif completion.finish_reason == "length":
        # Hit max_tokens limit
        # Output may be truncated mid-sentence
        return "Truncated at max_tokens"

    elif completion.finish_reason is None:
        # Generation incomplete (streaming context)
        return "Still generating"

    else:
        # Other reasons (abort, error)
        return f"Unknown: {completion.finish_reason}"
</syntaxhighlight>

'''Multi-Completion Handling:'''

When `n > 1`, multiple completions are returned:

<syntaxhighlight lang="python">
# Handling multiple completions
def select_best_completion(request_output):
    # By cumulative log probability (highest = most likely)
    best = max(
        request_output.outputs,
        key=lambda c: c.cumulative_logprob or float('-inf')
    )
    return best

def select_diverse_completions(request_output, k=3):
    # Return k most different completions
    # (requires custom similarity metric)
    return sorted(
        request_output.outputs,
        key=lambda c: diversity_score(c),
        reverse=True
    )[:k]
</syntaxhighlight>

'''Token-to-Text Alignment:'''

Token IDs and text may not align 1:1 due to subword tokenization:

<syntaxhighlight lang="python">
# Example: "Hello" might be one token but "unbelievable" might be 3
tokens = [15496, 318, 617, 2420]  # "This is some text"
# tokens don't map 1:1 to words

# Use tokenizer for accurate mapping
tokenizer = llm.get_tokenizer()
for token_id in completion.token_ids:
    token_str = tokenizer.decode([token_id])
    print(f"{token_id} -> '{token_str}'")
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_RequestOutput_usage]]
