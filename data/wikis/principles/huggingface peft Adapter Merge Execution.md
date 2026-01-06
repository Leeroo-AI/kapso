{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|TIES|https://arxiv.org/abs/2306.01708]]
* [[source::Paper|DARE|https://arxiv.org/abs/2311.03099]]
* [[source::Paper|Task Arithmetic|https://arxiv.org/abs/2212.04089]]
|-
! Domains
| [[domain::Model_Merging]], [[domain::Multi_Task]], [[domain::Adapter]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Principle for combining multiple adapters into a single merged adapter using advanced merging algorithms.

=== Description ===

Adapter Merge Execution combines task vectors (adapter deltas) using algorithms that handle:
* Weight interpolation with arbitrary weights
* Sign conflicts between adapters (TIES)
* Sparsification for robustness (DARE)
* Rank reduction via SVD

Different algorithms suit different scenarios—linear for simple interpolation, TIES for conflicting adapters, DARE for robust merging.

=== Usage ===

Apply this to combine capabilities from multiple trained adapters:
* **Linear:** Simple weighted average, fast, may average out conflicts
* **TIES:** Resolves sign conflicts, good for diverse adapters
* **DARE-TIES:** Adds random sparsification, more robust
* **SVD:** Controls output rank, memory efficient

== Theoretical Basis ==

'''Task Arithmetic (Linear):'''

Simple weighted sum of task vectors:
<math>\Delta W_{merged} = \sum_i w_i \cdot \Delta W_i</math>

Where <math>\Delta W_i = B_i A_i \cdot \text{scaling}_i</math>

'''TIES (TrIm, Elect Sign, Merge):'''

1. **Trim:** Keep top-k% by magnitude
2. **Elect Sign:** Determine majority sign per parameter
3. **Merge:** Sum only agreeing signs

<syntaxhighlight lang="python">
# Pseudo-code for TIES
def ties_merge(task_tensors, weights, density):
    # 1. Trim: Keep top density% parameters
    trimmed = [magnitude_prune(t, density) for t in task_tensors]

    # 2. Elect: Find majority sign
    signs = torch.stack([t.sign() for t in trimmed])
    majority_sign = signs.sum(dim=0).sign()

    # 3. Merge: Only include agreeing signs
    mask = (signs == majority_sign)
    merged = (trimmed * weights * mask).sum(dim=0)

    return merged / mask.sum(dim=0).clamp(min=1)
</syntaxhighlight>

'''DARE (Drop And REscale):'''

Random pruning with rescaling:
<math>\Delta W_{dare} = \frac{\text{Bernoulli}(p) \cdot \Delta W}{p}</math>

Reduces sensitivity to specific parameters while maintaining expected value.

'''SVD Rank Reduction:'''

Reduce merged adapter rank via SVD:
<syntaxhighlight lang="python">
def svd_merge(delta_w, target_rank):
    U, S, V = torch.linalg.svd(delta_w)
    U_r = U[:, :target_rank]
    S_r = S[:target_rank]
    V_r = V[:target_rank, :]

    # New A and B matrices
    A_new = (torch.diag(S_r.sqrt()) @ V_r)
    B_new = (U_r @ torch.diag(S_r.sqrt()))

    return A_new, B_new
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_peft_add_weighted_adapter]]
