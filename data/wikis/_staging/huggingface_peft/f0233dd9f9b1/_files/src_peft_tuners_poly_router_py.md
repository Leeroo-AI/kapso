# File: `src/peft/tuners/poly/router.py`

**Category:** component

| Property | Value |
|----------|-------|
| Lines | 82 |
| Classes | `Router`, `PolyRouter` |
| Functions | `get_router` |
| Imports | abc, config, torch |

## Understanding

**Status:** ✅ Fully explored

**Purpose:** Implements routing mechanisms that learn how to combine multiple LoRA modules for different tasks in Poly adapters.

**Mechanism:**
- **Router (ABC)**: Abstract base class defining the router interface
  - `reset()`: Abstract method for parameter initialization
  - `forward(task_ids, input_ids)`: Abstract method for computing routing weights

- **PolyRouter class**: Concrete implementation of task-based routing
  - **Parameters:**
    - `module_logits`: (n_tasks, n_splits * n_skills) learnable logits for each task
    - Shape allows each task to have different routing preferences
  - **Initialization:**
    - `reset()`: Initializes logits uniformly in range [-1e-3, 1e-3] (near zero)

  - **Forward Pass:**
    1. Validates task_ids exist and are in valid range [0, n_tasks)
    2. Indexes module_logits by task_ids to get task-specific logits
    3. Reshapes to (batch_size, n_splits, n_skills)
    4. **Training mode**: Applies Gumbel-Softmax via RelaxedBernoulli(temperature=1.0) for differentiable sampling
    5. **Inference mode**: Applies sigmoid activation
    6. Normalizes across skills dimension to get routing weights that sum to 1

  - **Output:** Mixing weights of shape (batch_size, n_splits, n_skills)

- **get_router() function**: Factory function that returns appropriate router based on poly_type (currently only supports "poly")

- **EPS constant (1e-12)**: Small epsilon value to prevent division by zero during normalization

**Significance:** Critical component that enables Poly's multi-task learning capability. The router learns task-specific combinations of shared LoRA modules:
1. Each task gets its own set of logits to control skill mixing
2. Gumbel-Softmax during training allows gradient flow through discrete decisions
3. Soft routing weights during inference enable smooth skill blending
4. The design is inspired by the Polytropon paper's multi-task adapter routing mechanism
The router effectively learns a different "recipe" for combining the shared LoRA skills for each task, enabling efficient parameter sharing while maintaining task-specific adaptation.
