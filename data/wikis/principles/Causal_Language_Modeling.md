{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|GPT-2|https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf]]
* [[source::Paper|LLaMA|https://arxiv.org/abs/2302.13971]]
* [[source::Doc|HuggingFace Causal LM|https://huggingface.co/docs/transformers/tasks/language_modeling]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::NLP]], [[domain::Language_Modeling]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Language modeling paradigm where models predict the next token based only on previous tokens, using left-to-right autoregressive generation.

=== Description ===
Causal Language Modeling (CLM) is the training objective used by GPT, Llama, Mistral, and most modern LLMs. The model learns to predict each token given all previous tokens, using a causal attention mask to prevent looking at future tokens. This autoregressive structure enables open-ended text generation and forms the foundation of instruction-following capabilities.

=== Usage ===
Use this principle as the foundation for understanding how LLMs generate text. Causal LM is the base training objective before instruction tuning. Understanding CLM is essential for working with decoder-only transformers, implementing generation strategies, and debugging output quality issues.

== Theoretical Basis ==
'''Training Objective:'''
Maximize the likelihood of each token given its left context:

\[
\mathcal{L}_{CLM} = -\sum_{t=1}^{T} \log P_\theta(x_t | x_{<t})
\]

Where \(x_{<t} = (x_1, x_2, ..., x_{t-1})\).

'''Causal Attention Mask:'''
Prevents attending to future tokens:

\[
\text{Mask}_{i,j} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}
\]

'''Generation Process:'''
<syntaxhighlight lang="python">
def generate_causal(model, prompt, max_tokens):
    """
    Autoregressive generation
    """
    tokens = tokenize(prompt)
    
    for _ in range(max_tokens):
        # Get logits for next token
        logits = model(tokens)[:, -1, :]  # Last position only
        
        # Sample or argmax
        next_token = sample(logits)
        
        # Append and continue
        tokens = concat(tokens, next_token)
        
        if next_token == EOS:
            break
    
    return tokens
</syntaxhighlight>

'''Key Properties:'''
1. **Unidirectional**: Only sees past, not future
2. **Autoregressive**: Generates one token at a time
3. **No padding needed**: Natural handling of variable length
4. **Caching**: KV cache enables efficient generation

'''Causal vs Masked LM:'''
{| class="wikitable"
! Aspect !! Causal LM (GPT) !! Masked LM (BERT)
|-
|| Direction || Left-to-right || Bidirectional
|-
|| Mask || Causal (triangular) || Random positions
|-
|| Use case || Generation || Understanding
|-
|| Examples || GPT, Llama, Mistral || BERT, RoBERTa
|}

== Related Pages ==
=== Implemented By ===
* [[implemented_by::Implementation:Unsloth_FastModel]]

=== Tips and Tricks ===
(Fundamental concept - no specific heuristics)

