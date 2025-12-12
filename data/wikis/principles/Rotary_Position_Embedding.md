{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|RoFormer: Enhanced Transformer with Rotary Position|https://arxiv.org/abs/2104.09864]]
* [[source::Blog|Rotary Position Embeddings|https://blog.eleuther.ai/rotary-embeddings/]]
* [[source::Paper|LLaMA Paper|https://arxiv.org/abs/2302.13971]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::NLP]], [[domain::Position_Encoding]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Position encoding technique that applies rotations to query and key vectors, enabling relative position awareness and seamless context length extension.

=== Description ===
Rotary Position Embedding (RoPE) encodes position information by rotating the query and key vectors in attention. Unlike absolute position embeddings, RoPE naturally captures relative positions through the geometric properties of rotations. This enables better length generalization and is the position encoding used in Llama, Mistral, Qwen, and most modern LLMs.

=== Usage ===
Use this principle when understanding how position information is encoded in modern LLMs. Critical for extending context length beyond training length (RoPE scaling). Understanding RoPE helps when debugging position-related issues or implementing context extension techniques like NTK-aware scaling or YaRN.

== Theoretical Basis ==
'''Core Idea:'''
Rotate query and key vectors by an angle proportional to their position:

\[
\text{RoPE}(x_m, m) = x_m \cdot e^{im\theta}
\]

Where m is position and θ is a learned or fixed frequency.

'''2D Rotation Matrix Form:'''
For each pair of dimensions:
\[
R_\theta^m = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}
\]

'''Key Property - Relative Position:'''
\[
\langle \text{RoPE}(q, m), \text{RoPE}(k, n) \rangle = \langle q, k \rangle \cdot f(m-n)
\]

The dot product only depends on relative position (m-n), not absolute positions.

'''Implementation:'''
<syntaxhighlight lang="python">
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """
    q, k: (batch, heads, seq_len, head_dim)
    cos, sin: Precomputed rotation matrices
    """
    # Split into pairs of dimensions
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    
    # Apply rotation
    q_rotated = torch.cat([q1 * cos - q2 * sin,
                           q1 * sin + q2 * cos], dim=-1)
    k_rotated = torch.cat([k1 * cos - k2 * sin,
                           k1 * sin + k2 * cos], dim=-1)
    
    return q_rotated, k_rotated
</syntaxhighlight>

'''Context Extension:'''
RoPE naturally supports length extension through frequency scaling:
* Linear scaling: θ' = θ / scale
* NTK-aware: Adjust base frequency
* YaRN: Learned attention scaling

== Related Pages ==
=== Implemented By ===
* [[implemented_by::Implementation:Unsloth_FastLanguageModel]]

=== Tips and Tricks ===
(Position encoding is handled automatically by Unsloth)

