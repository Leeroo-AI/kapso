{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::Structured_Output]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

The process of generating text that conforms to specified constraints through logit masking and guided decoding.

=== Description ===

Constrained Generation ensures outputs match specified patterns or schemas:

1. **Logit Masking:** Invalid tokens get -inf logits
2. **State Tracking:** FSM/parser tracks valid continuations
3. **Efficient Sampling:** Only valid tokens considered
4. **Transparent API:** Same generate() interface
5. **Batch Support:** Multiple constrained requests in parallel

=== Usage ===

Use constrained generation when:
- Extracting structured data from text
- Building function calling systems
- Implementing classification pipelines
- Ensuring output format compliance

== Theoretical Basis ==

'''Constraint Enforcement Flow:'''

<syntaxhighlight lang="text">
Input: prompt + constraint
                 │
                 ▼
┌────────────────────────────────────┐
│ Model Forward Pass                  │
│ logits = model(input_ids)           │
└────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│ Constraint Processor                │
│ valid_tokens = get_valid(state)     │
│ logits[~valid] = -inf               │
└────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│ Sampling                            │
│ token = sample(logits, temp, top_p) │
└────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│ State Update                        │
│ state = advance(state, token)       │
└────────────────────────────────────┘
                 │
                 ▼
         Repeat until done
</syntaxhighlight>

'''Logit Masking:'''

<syntaxhighlight lang="python">
def apply_constraint(logits: torch.Tensor, valid_tokens: set[int]):
    """Mask invalid tokens to prevent sampling."""
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[list(valid_tokens)] = False
    logits[mask] = float("-inf")
    return logits
</syntaxhighlight>

'''Constraint State Machine (JSON example):'''

<syntaxhighlight lang="text">
JSON Object Generation:
State 0: Expect '{'
State 1: Expect '"' (key) or '}'
State 2: Expect key characters
State 3: Expect '"' (end key)
State 4: Expect ':'
State 5: Expect value
State 6: Expect ',' or '}'
...
</syntaxhighlight>

'''Backend Integration:'''

<syntaxhighlight lang="python">
# vLLM supports multiple backends
backends = {
    "outlines": "FSM-based, supports json/regex/grammar",
    "lm-format-enforcer": "Alternative JSON/regex support",
    "xgrammar": "Grammar-based constraints",
}

# Backend selection is automatic based on constraint type
</syntaxhighlight>

'''Performance Considerations:'''

<syntaxhighlight lang="text">
Factor              Impact
─────────────────   ─────────────────────────────────
Schema complexity   Larger schemas = more FSM states
Vocab size          Larger vocab = more masking
Batch size          Constraints per request
Token count         More tokens = more state transitions
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_LLM_generate_structured]]
