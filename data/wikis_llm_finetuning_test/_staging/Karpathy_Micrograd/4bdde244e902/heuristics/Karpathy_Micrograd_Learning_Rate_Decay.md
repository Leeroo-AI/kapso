# Heuristic: Learning_Rate_Decay

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|micrograd demo|https://github.com/karpathy/micrograd/blob/master/demo.ipynb]]
* [[source::Blog|Learning Rate Schedules|https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-13 12:00 GMT]]
|}

== Overview ==

Linear learning rate decay from 1.0 to 0.1 over training for stable convergence with SGD.

=== Description ===

The micrograd demo uses a simple linear learning rate schedule that starts at 1.0 and decays to 0.1 over 100 training steps. This aggressive initial learning rate allows rapid early progress, while the decay ensures stable convergence as training progresses and the loss landscape requires finer adjustments.

The formula `lr = 1.0 - 0.9 * (step / total_steps)` provides a deterministic, predictable decay without requiring external schedulers or complex hyperparameter tuning.

=== Usage ===

Apply this learning rate schedule when training small networks with SGD. Start with a high learning rate for fast initial convergence, then decay to prevent oscillation around the minimum. Adjust the initial rate (1.0) and final rate (0.1) based on your specific problem - these values work well for the 2D classification demo but may need tuning for other tasks.

== The Insight (Rule of Thumb) ==

* **Action:** Use linear decay: `learning_rate = initial_lr * (1 - decay_factor * step / total_steps)`
* **Value:** Initial LR: 1.0, Final LR: 0.1 (10x decay over training)
* **Trade-off:** Simple schedule; may not be optimal for all problems. High initial LR may cause instability on harder problems.

== Reasoning ==

Large learning rates early in training allow escaping poor local minima and making rapid progress. As training progresses and the model approaches a good solution, smaller learning rates enable fine-grained adjustments without overshooting.

The micrograd demo achieves 100% accuracy on the moons dataset within ~40 steps with this schedule. The loss curve shows characteristic rapid descent followed by gradual refinement:
- Steps 0-20: Loss drops from 0.9 to 0.2 (80% reduction)
- Steps 20-100: Loss stabilizes around 0.01 (further 95% reduction)

== Code Evidence ==

Learning rate schedule from `demo.ipynb` cell 7:
<syntaxhighlight lang="python">
# optimization
for k in range(100):

    # forward
    total_loss, acc = loss()

    # backward
    model.zero_grad()
    total_loss.backward()

    # update (sgd)
    learning_rate = 1.0 - 0.9*k/100
    for p in model.parameters():
        p.data -= learning_rate * p.grad
</syntaxhighlight>

Note the manual parameter update loop - micrograd doesn't have a built-in optimizer, so learning rate scheduling is applied directly in the training loop.

== Related Pages ==

* [[used_by::Implementation:Karpathy_Micrograd_Module_Parameters]]
* [[used_by::Principle:Karpathy_Micrograd_Parameter_Update]]
* [[used_by::Principle:Karpathy_Micrograd_Training_Loop]]
