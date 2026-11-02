# Implementation: Distillation_Loss_KL

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PyTorch|https://github.com/pytorch/pytorch]]
* [[source::Doc|torch.nn.functional.kl_div|https://pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Model_Compression]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Student loss for knowledge distillation computed with torch.nn.functional.kl_div on temperature-softened logits, rescaled by the squared temperature.

=== Description ===

kl_div expects log-probabilities as input and probabilities as target, so the student logits go through log_softmax and the teacher logits through softmax, both after division by the temperature.

=== Usage ===

Add it to the label loss in the student's training step.

== Code Reference ==

<syntaxhighlight lang="python">
import torch.nn.functional as F
def distill_loss(student_logits, teacher_logits, T=2.0):
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * T * T
</syntaxhighlight>
