{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LoRA: Low-Rank Adaptation|https://arxiv.org/abs/2106.09685]]
* [[source::Paper|QLoRA|https://arxiv.org/abs/2305.14314]]
* [[source::Blog|HuggingFace PEFT|https://huggingface.co/docs/peft]]
|-
! Domains
| [[domain::Fine_Tuning]], [[domain::PEFT]], [[domain::Deep_Learning]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Parameter-efficient fine-tuning technique that adapts pre-trained models by injecting trainable low-rank decomposition matrices into existing layers.

=== Description ===
Low-Rank Adaptation (LoRA) is a technique for fine-tuning large language models efficiently. Instead of updating all model parameters, LoRA freezes the pre-trained weights and injects small trainable matrices into each layer. These matrices are decomposed into two low-rank components (A and B), dramatically reducing the number of trainable parameters. For a weight matrix W, the update is W' = W + BA, where B and A have much lower rank than W.

=== Usage ===
Use this principle when you need to fine-tune large language models (7B+ parameters) but have limited GPU memory or storage. Ideal for instruction tuning, domain adaptation, and style transfer tasks. Prefer LoRA over full fine-tuning when you want to maintain multiple task-specific adapters that can be swapped at inference time.

== Theoretical Basis ==
The core insight is that the weight updates during fine-tuning have low "intrinsic rank" - they can be approximated by low-rank matrices without significant loss.

For a pre-trained weight matrix \( W_0 \in \mathbb{R}^{d \times k} \):

\[
W = W_0 + \Delta W = W_0 + BA
\]

Where:
* \( B \in \mathbb{R}^{d \times r} \)
* \( A \in \mathbb{R}^{r \times k} \)  
* \( r \ll \min(d, k) \) is the rank

'''Parameter Reduction:'''
* Full fine-tuning: \( d \times k \) parameters
* LoRA: \( r \times (d + k) \) parameters
* For d=k=4096, r=16: 99.6% reduction

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
class LoRALayer:
    def __init__(self, d, k, r, alpha):
        self.W = frozen_pretrained_weight  # (d, k)
        self.A = nn.Parameter(torch.randn(r, k))  # Trainable
        self.B = nn.Parameter(torch.zeros(d, r))  # Trainable
        self.scaling = alpha / r
    
    def forward(self, x):
        # Original computation + low-rank update
        return self.W @ x + self.scaling * (self.B @ (self.A @ x))
</syntaxhighlight>

== Related Pages ==
=== Implemented By ===
* [[implemented_by::Implementation:Unsloth_get_peft_model]]

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:LoRA_Rank_Selection]]
* [[uses_heuristic::Heuristic:QLoRA_Target_Modules_Selection]]

