{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Attention Is All You Need|https://arxiv.org/abs/1706.03762]]
* [[source::Blog|Illustrated Transformer|https://jalammar.github.io/illustrated-transformer/]]
* [[source::Paper|BERT|https://arxiv.org/abs/1810.04805]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::NLP]], [[domain::Attention]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Mechanism that allows neural networks to weigh the importance of different input positions dynamically based on their relevance to each other.

=== Description ===
Self-Attention is the core mechanism of Transformer architectures. It computes attention scores between all pairs of positions in a sequence, allowing the model to capture long-range dependencies regardless of distance. Unlike RNNs which process sequentially, self-attention processes all positions in parallel, enabling massive scalability. It's the fundamental building block of modern LLMs like GPT, Llama, and Mistral.

=== Usage ===
Use this principle when designing architectures for sequence modeling tasks (NLP, time series) where capturing long-term context is critical. It is the fundamental building block of all modern Large Language Models. Understanding self-attention is essential for working with Transformers, debugging attention patterns, and optimizing memory usage during training.

== Theoretical Basis ==
The core operation is scaled dot-product attention:

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

Where:
* Q (Query): What we're looking for - shape (seq_len, d_k)
* K (Key): What we match against - shape (seq_len, d_k)
* V (Value): What we retrieve - shape (seq_len, d_v)
* d_k: Key dimension (scaling factor prevents softmax saturation)

'''Multi-Head Attention:'''
Multiple attention heads capture different relationship types:
\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
\]

'''Pseudo-code:'''
<syntaxhighlight lang="python">
def self_attention(x, W_q, W_k, W_v, mask=None):
    """
    x: input sequence (batch, seq_len, d_model)
    """
    Q = x @ W_q  # (batch, seq_len, d_k)
    K = x @ W_k
    V = x @ W_v
    
    # Attention scores
    scores = Q @ K.transpose(-2, -1) / sqrt(d_k)
    
    # Apply causal mask for autoregressive models
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    weights = softmax(scores, dim=-1)
    output = weights @ V
    
    return output
</syntaxhighlight>

'''Memory Complexity:'''
* Attention matrix: O(n²) where n = sequence length
* This is why long-context models need FlashAttention optimization

== Related Pages ==
=== Implemented By ===
* [[implemented_by::Implementation:Unsloth_FastLanguageModel]]

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:Memory_Efficient_Attention]]

