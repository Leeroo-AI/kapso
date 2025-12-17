# Poly Configuration

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/poly/config.py`
- **Lines**: 103
- **Purpose**: Configuration for Poly (Polytropon) multi-skill LoRA models

## Overview

PolyConfig defines configuration for Poly adapters that use multiple LoRA-like "skills" with learned routing to combine them. Supports both Polytropon and Multi-Head Routing (MHR) variants.

## PolyConfig Class

**Inheritance**: Extends `PeftConfig`

### Configuration Parameters

1. **`r` (int, default=8)**
   - LoRA rank for each skill
   - Same meaning as standard LoRA
   - Applied to all skills

2. **`target_modules` (Union[List[str], str], optional)**
   - Modules to apply Poly to
   - Examples: `['q', 'v']`, `'.*decoder.*'`

3. **`exclude_modules` (Union[List[str], str], optional)**
   - Modules to exclude from Poly
   - Useful with wildcard target_modules

4. **`modules_to_save` (List[str], optional)**
   - Additional modules to train beyond Poly layers

5. **`init_weights` (bool, default=True)**
   - If True: Initialize B matrices to zeros (LoRA-style)
   - If False: Kaiming initialization (for testing)

6. **`poly_type` (Literal["poly"], default="poly")**
   - Type of Poly module
   - Currently only "poly" supported
   - Reserved for future variants

7. **`n_tasks` (int, default=1)**
   - Number of tasks in multitasking scenario
   - Router learns separate weights for each task
   - Must be ≥1

8. **`n_skills` (int, default=4)**
   - Number of skills (LoRA modules) per layer
   - Each skill is a separate low-rank adaptation
   - Higher values = more capacity but more parameters

9. **`n_splits` (int, default=1)**
   - Number of splits for Multi-Head Routing (MHR)
   - If 1: Standard Polytropon
   - If >1: MHR with independent routing per split
   - Must divide layer dimensions

### Post-Initialization

```python
def __post_init__(self):
    super().__post_init__()
    self.peft_type = PeftType.POLY

    # Convert to sets
    self.target_modules = (
        set(self.target_modules) if isinstance(self.target_modules, list)
        else self.target_modules
    )
    self.exclude_modules = (
        set(self.exclude_modules) if isinstance(self.exclude_modules, list)
        else self.exclude_modules
    )
```

## Configuration Patterns

### Basic Polytropon
```python
from peft import PolyConfig

config = PolyConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    n_tasks=10,      # 10 different tasks
    n_skills=4,      # 4 skills per layer
    n_splits=1       # Standard Polytropon
)
```

### Multi-Head Routing (MHR)
```python
config = PolyConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    n_tasks=10,
    n_skills=4,
    n_splits=4       # MHR with 4 heads
)
```

### Large-Scale Multi-Task
```python
config = PolyConfig(
    r=16,
    target_modules="all-linear",
    exclude_modules=["lm_head"],
    n_tasks=100,     # Many tasks
    n_skills=8,      # More skills for capacity
    n_splits=1
)
```

### Minimal Configuration
```python
config = PolyConfig(
    r=4,
    target_modules=["q_proj", "v_proj"],
    n_tasks=5,
    n_skills=2,      # Fewer skills for efficiency
    n_splits=1
)
```

## Parameter Analysis

### Polytropon (n_splits=1)

For layer d×d, per adapter:
- **Skills**: n_skills × r × (2d)
- **Router**: n_tasks × n_skills

**Example** (d=4096, r=8, n_tasks=10, n_skills=4):
- Skills: 4 × 8 × 8192 = 262,144
- Router: 10 × 4 = 40
- **Total**: 262,184

### Multi-Head Routing (n_splits=4)

Same example with n_splits=4:
- Skills: Same (262,144)
- Router: 10 × 4 × 4 = 160
- **Total**: 262,304

MHR adds minimal overhead (120 params) for routing flexibility.

### Comparison with LoRA

**LoRA** (r=8, single task):
- Parameters: 8 × 8192 = 65,536

**Poly** (r=8, n_skills=4, n_tasks=10):
- Parameters: 262,184
- **4x more than LoRA** but supports 10 tasks with 4 skills

**Per-Task View**:
- Poly: 262,184 params for 10 tasks = 26,218 per task
- LoRA: 65,536 params per task
- **Poly is 2.5x more efficient per task**

## Design Considerations

### 1. Number of Skills (n_skills)

**Fewer Skills** (2-4):
- Less memory
- Faster computation
- May limit expressiveness
- Good for similar tasks

**More Skills** (8-16):
- More capacity
- Better for diverse tasks
- More parameters
- Richer skill library

### 2. Number of Splits (n_splits)

**Single Split** (n_splits=1):
- Standard Polytropon
- Simpler
- Less routing overhead

**Multiple Splits** (n_splits=2,4,8):
- Multi-Head Routing (MHR)
- More flexible
- Better for complex tasks
- Slight parameter increase
- Must divide layer dimensions

### 3. Number of Tasks (n_tasks)

**Few Tasks** (1-10):
- Simple routing
- Easy to manage

**Many Tasks** (50-100+):
- Router parameters grow: n_tasks × n_splits × n_skills
- Still efficient compared to per-task LoRA
- Enables large-scale multitasking

### 4. Rank Selection (r)

**Small Rank** (4-8):
- Efficient per skill
- Good for most tasks

**Large Rank** (16-32):
- More capacity per skill
- Better for complex adaptations
- More parameters per skill

## Task-Specific Considerations

### Similar Tasks
```python
config = PolyConfig(
    r=8,
    n_tasks=10,
    n_skills=2,      # Few skills, similar adaptations
    n_splits=1
)
```

### Diverse Tasks
```python
config = PolyConfig(
    r=8,
    n_tasks=10,
    n_skills=8,      # Many skills, diverse adaptations
    n_splits=4       # MHR for fine-grained control
)
```

### Single Task with Specialization
```python
config = PolyConfig(
    r=8,
    n_tasks=1,       # Just one task
    n_skills=4,      # Multiple skills for different aspects
    n_splits=1
)
```

## Validation Rules

1. **r > 0**: Rank must be positive
2. **n_tasks >= 1**: At least one task
3. **n_skills >= 1**: At least one skill (typically ≥2 for multi-skill)
4. **n_splits >= 1**: At least one split
5. **n_splits divides dimensions**: n_splits must divide layer dimensions

## Integration with PEFT

```python
from transformers import AutoModelForCausalLM
from peft import get_peft_model, PolyConfig

base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

config = PolyConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    n_tasks=10,
    n_skills=4,
    n_splits=1
)

model = get_peft_model(base_model, config)

# Must provide task_ids during forward
task_ids = torch.tensor([0, 1, 2, 0])  # Batch of 4
output = model(input_ids, task_ids=task_ids)
```

## Configuration Storage

```json
{
  "peft_type": "POLY",
  "r": 8,
  "target_modules": ["q_proj", "v_proj"],
  "exclude_modules": null,
  "modules_to_save": null,
  "init_weights": true,
  "poly_type": "poly",
  "n_tasks": 10,
  "n_skills": 4,
  "n_splits": 1
}
```

## Best Practices

1. **Start Simple**: n_skills=4, n_splits=1
2. **Scale Up**: Increase n_skills for diverse tasks
3. **Use MHR**: Try n_splits=2 or 4 for complex scenarios
4. **Monitor Router**: Check which skills are being used
5. **Task IDs**: Ensure correct task_ids during training/inference

## Advanced Configurations

### Hierarchical Skills
```python
# Layer 1: General skills
config_l1 = PolyConfig(r=8, n_skills=4, n_tasks=10)

# Layer 2: Specialized skills
config_l2 = PolyConfig(r=16, n_skills=8, n_tasks=10)
```

### Progressive Training
```python
# Stage 1: Few skills
config_stage1 = PolyConfig(n_skills=2, n_tasks=5)

# Stage 2: Add more skills and tasks
config_stage2 = PolyConfig(n_skills=4, n_tasks=10)
```

## Limitations

1. **Task ID Required**: Must provide task_ids at inference
2. **Memory**: n_skills × LoRA memory
3. **Single poly_type**: Only "poly" currently supported
4. **No Nested Adapters**: Cannot combine with other PEFT methods easily

## References

- **Polytropon**: https://huggingface.co/papers/2202.13914
- **Multi-Head Routing**: https://huggingface.co/papers/2211.03831
- **Concept**: Multi-task learning with shared skill library
