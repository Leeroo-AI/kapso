# Heuristic: unslothai_unsloth_RL_Learning_Rate

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Paper|GRPO|https://arxiv.org/abs/2402.03300]]
* [[source::Discussion|TRL Issues|https://github.com/huggingface/trl/issues]]
|-
! Domains
| [[domain::Reinforcement_Learning]], [[domain::GRPO]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2025-12-16 12:00 GMT]]
|}

## Overview

Use much lower learning rates for RL/GRPO training (5e-6) compared to SFT (2e-4) to prevent policy collapse.

### Description

Reinforcement learning fine-tuning (GRPO, PPO, DPO) requires significantly lower learning rates than supervised fine-tuning (SFT):

- **SFT**: Typical LR = 2e-4 to 5e-5
- **RL**: Typical LR = 5e-6 to 2e-5

Higher learning rates in RL cause:
1. **Policy collapse**: Model degenerates to repetitive outputs
2. **Reward hacking**: Model finds shortcuts that maximize reward without quality
3. **Instability**: Training loss oscillates wildly

### Usage

Use this heuristic when:
- Setting up GRPO or other RL training
- Debugging unstable RL training
- Observing policy collapse symptoms

## The Insight (Rule of Thumb)

* **Action:** Set learning rate 10-40x lower for RL than SFT
* **SFT Baseline:** `learning_rate=2e-4`
* **RL Recommended:** `learning_rate=5e-6` to `1e-5`
* **Trade-off:** Slower convergence but stable training

### Learning Rate Guidelines

| Training Type | Learning Rate | Notes |
|---------------|---------------|-------|
| SFT | 2e-4 | Standard fine-tuning |
| DPO | 5e-6 | Conservative |
| GRPO | 5e-6 to 1e-5 | Depends on reward scale |
| PPO | 1e-6 to 5e-6 | Most sensitive |

### Signs of Wrong LR

**Too High:**
- Loss spikes or NaN
- Model outputs become repetitive
- Reward collapses after initial increase

**Too Low:**
- No improvement over many steps
- Reward stays flat
- Model unchanged from base

## Reasoning

In SFT, we directly maximize likelihood of target tokens - a convex-ish objective with clear gradients.

In RL, we optimize a policy via reward signals:
1. Policy changes affect reward distribution
2. Reward distribution affects gradient estimates
3. Gradient estimates affect policy

This feedback loop amplifies any instability. Small policy changes should produce small reward changes, but large LR causes large policy jumps that completely change the reward landscape.

## Code Evidence

From `unsloth/models/rl.py` torch compile options:
```python
torch_compile_options = {
    "epilogue_fusion": True,
    "max_autotune": False,  # Disable Triton mm kernels - stability
    "shape_padding": True,
    "trace.enabled": False,
    "triton.cudagraphs": False,  # Disabled for stability
}
```

GRPO configuration typically includes:
```python
GRPOConfig(
    learning_rate=5e-6,  # Much lower than SFT
    # ...
)
```

## Related Pages

* [[uses_heuristic::Workflow:unslothai_unsloth_GRPO_Training]]
* [[uses_heuristic::Principle:unslothai_unsloth_GRPO_Training]]
