# Implementation: adaption_prompt/layer.py

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/adaption_prompt/layer.py`
- **Size**: 236 lines
- **Description**: Gated attention mechanism with learnable prompts

## Overview

This module implements the core attention adaptation mechanism, wrapping existing attention layers and injecting gated prompt tokens that can attend to the input without interfering with the base model initially.

## Core Classes

### _BaseAdaptedAttention

**Initialization**:
```python
self.adaption_prompt = nn.Parameter(torch.empty(1, adapter_len, hidden_size).normal_())
self.adaption_gate = nn.Parameter(torch.zeros(1))  # Zero-init gating
```

**Purpose**: Inserts adapter_len prompt tokens into attention K/V

### AdaptedAttentionGPT (for GPT-2)

**Forward Process**:
1. Run base attention
2. Project adaption_prompt through c_attn → K, V
3. Repeat for batch_size
4. Recompute Q from hidden_states
5. Compute attention: `scores = Q @ adapter_K^T / sqrt(head_dim)`
6. Apply softmax and gate: `adapter_output = gate * softmax(scores) @ adapter_V`
7. Add to base output

### AdaptedAttention (for LLaMA/Mistral)

**Similar to GPT but**:
- Uses separate k_proj/v_proj
- Handles GQA (grouped query attention)
- Includes o_proj for output projection

## Key Features

### Zero-Init Gating

```python
self.adaption_gate = nn.Parameter(torch.zeros(1))
```

**Benefit**: Model starts identical to base model, gradually learns to use prompts

### Efficient Prompt Injection

**No Position Encoding**: Prompts are position-independent
**Parallel Computation**: Adapter attention computed alongside base attention

## Cross-References

- **Config**: `adaption_prompt/config.py`
- **Model**: `adaption_prompt/model.py`
