# File: `src/peft/tuners/poly/layer.py`

**Category:** layer

| Property | Value |
|----------|-------|
| Lines | 166 |
| Classes | `PolyLayer`, `Linear` |
| Imports | config, math, peft, router, torch, typing |

## Understanding

**Status:** ✅ Fully explored

**Purpose:** Implements Poly adapter layers that combine multiple LoRA modules with learned routing for multi-task learning.

**Mechanism:**
- **PolyLayer class** (base layer):
  - Stores adapter parameters: `poly_lora_A`, `poly_lora_B` (ParameterDict), `poly_router` (ModuleDict)
  - Tracks metadata: `r`, `n_tasks`, `n_skills`, `n_splits` (all dicts keyed by adapter name)
  - `adapter_layer_names = ("poly_lora_A", "poly_lora_B", "poly_router")`: Trainable adapter components
  - `update_layer()`: Initializes LoRA matrices and router for an adapter
  - `reset_poly_parameters()`: Initializes weights using Kaiming uniform for A, zeros for B (or Kaiming for both if init_weights=False)

- **LoRA Matrices Shape:**
  - `poly_lora_A`: (n_splits, n_skills, in_features // n_splits, r)
  - `poly_lora_B`: (n_splits, n_skills, r, out_features // n_splits)
  - Splits enable Multi-Head Routing (MHR) when n_splits > 1

- **Linear class** (concrete implementation):
  - Inherits from both `nn.Module` and `PolyLayer`
  - `forward()` method:
    1. Runs base layer forward pass
    2. For each active adapter:
       - Gets routing weights from `poly_router(task_ids, input_ids)`
       - Computes mixed LoRA matrices using einsum operations
       - Applies low-rank adaptation: `result += x @ A @ B / r`
    3. Returns result in original dtype

- **Routing Integration:**
  - Router produces mixing weights of shape (batch_size, n_splits, n_skills)
  - Uses einsum to combine LoRA matrices: weighted sum across skills dimension
  - Final computation: `x.bmm(A).bmm(B) / r` for batch matrix multiplication

**Significance:** Core implementation of Poly's multi-task learning mechanism. The key innovation is the learnable routing that combines multiple LoRA modules dynamically based on task IDs. This allows parameter-efficient multi-task learning where:
1. Multiple tasks share the same set of LoRA "skills"
2. Each task learns its own routing weights to combine these skills
3. Multi-Head Routing (n_splits > 1) provides finer-grained control
The design significantly reduces parameters compared to having separate LoRA adapters per task while maintaining task-specific adaptation capabilities.
