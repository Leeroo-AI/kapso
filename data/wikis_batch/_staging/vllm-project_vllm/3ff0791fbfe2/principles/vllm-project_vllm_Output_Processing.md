# Output Processing

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::Inference]], [[domain::Output_Processing]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:30 GMT]]
|}

== Overview ==

Principle for accessing and processing the structured outputs from LLM generation, including generated text, token IDs, log probabilities, and completion metadata.

=== Description ===

Output Processing defines how generation results are structured and accessed. vLLM provides rich output objects containing:

1. **Generated text**: The detokenized completion string
2. **Token IDs**: Raw token identifiers for post-processing
3. **Log probabilities**: Per-token probability information (when requested)
4. **Finish metadata**: Why generation stopped (EOS, stop string, length limit)
5. **Metrics**: Timing and performance statistics

The hierarchical output structure supports:
* Multiple completions per prompt (n > 1)
* Both streaming and batch modes
* Detailed analysis through logprobs
* Request-level and completion-level information

=== Usage ===

Process outputs when:
* Extracting generated text for downstream applications
* Analyzing generation quality through log probabilities
* Implementing custom post-processing logic
* Monitoring generation metrics and performance
* Debugging generation issues through token-level inspection

== Theoretical Basis ==

'''Output Structure Hierarchy:'''
<syntaxhighlight lang="python">
# Abstract output structure
RequestOutput:
    request_id: str           # Unique request identifier
    prompt: str               # Input prompt
    prompt_token_ids: list    # Tokenized prompt
    prompt_logprobs: list     # Prompt token probabilities (optional)
    outputs: list[CompletionOutput]  # One per n value
    finished: bool            # All completions done
    metrics: RequestMetrics   # Timing stats

CompletionOutput:
    index: int                # Which of n outputs
    text: str                 # Generated text
    token_ids: list[int]      # Generated tokens
    cumulative_logprob: float # Total log probability
    logprobs: list[dict]      # Per-token probabilities (optional)
    finish_reason: str        # "stop", "length", etc.
    stop_reason: str | int    # What triggered stop
</syntaxhighlight>

'''Finish Reasons:'''
* `"stop"`: Hit stop string or stop token
* `"length"`: Reached max_tokens limit
* `"abort"`: Request was cancelled

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_RequestOutput]]
