{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|FlashAttention|https://arxiv.org/abs/2205.14135]]
* [[source::Paper|FlashAttention-2|https://arxiv.org/abs/2307.08691]]
* [[source::Repo|FlashAttention GitHub|https://github.com/Dao-AILab/flash-attention]]
|-
! Domains
| [[domain::Optimization]], [[domain::Attention]], [[domain::Deep_Learning]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
IO-aware attention algorithm that reduces memory usage from O(n²) to O(n) while also improving speed through kernel fusion.

=== Description ===
FlashAttention revolutionizes attention computation by restructuring the algorithm to minimize memory reads/writes to GPU high-bandwidth memory (HBM). Instead of materializing the full NxN attention matrix, it computes attention in blocks using GPU SRAM (fast memory), achieving both memory efficiency and speed improvements. FlashAttention-2 further optimizes this with better parallelization and reduced non-matmul FLOPs.

=== Usage ===
Use this principle when training or inferencing with long sequences (4K+ tokens). Essential for extending context length of LLMs. Unsloth automatically uses FlashAttention when available, providing 12x longer context compared to standard attention implementations.

== Theoretical Basis ==
'''Standard Attention Memory Issue:'''
Computing \(\text{softmax}(QK^T)\) requires materializing an \(N \times N\) matrix, where N = sequence length.
* 8K sequence: 8192 × 8192 × 4 bytes = 256 MB per layer per batch
* Not scalable for long contexts

'''FlashAttention Algorithm:'''
Key insight: softmax can be computed incrementally without full materialization.

\[
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}
\]

Where m = max(x) for numerical stability.

'''Tiled Computation:'''
<syntaxhighlight lang="python">
def flash_attention_block(Q_block, K_blocks, V_blocks):
    """
    Process attention in tiles without full N×N matrix
    """
    O = zeros_like(Q_block)  # Output accumulator
    l = zeros(Q_block.shape[0])  # Normalizer accumulator
    m = full(Q_block.shape[0], -inf)  # Max accumulator
    
    for K_b, V_b in zip(K_blocks, V_blocks):
        # Compute attention scores for this block
        S = Q_block @ K_b.T / sqrt(d_k)
        
        # Online softmax update
        m_new = maximum(m, S.max(dim=-1))
        l = l * exp(m - m_new) + exp(S - m_new).sum(dim=-1)
        O = O * exp(m - m_new)[:, None] + exp(S - m_new) @ V_b
        m = m_new
    
    return O / l[:, None]
</syntaxhighlight>

'''Benefits:'''
{| class="wikitable"
! Aspect !! Standard !! FlashAttention
|-
|| Memory || O(N²) || O(N)
|-
|| HBM Accesses || O(N²) || O(N² / SRAM)
|-
|| Speed || Baseline || 2-4x faster
|}

== Related Pages ==
=== Implemented By ===
* [[implemented_by::Implementation:Unsloth_FastLanguageModel]]

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:Memory_Efficient_Attention]]

