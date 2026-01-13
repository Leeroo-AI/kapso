# Workflow Index: Karpathy_Micrograd

> Comprehensive index of Workflows and their implementation context.
> This index bridges Phase 1 (Anchoring) and Phase 2 (Repository Building).
> **Update IMMEDIATELY** after creating or modifying a Workflow page.

---

## Summary

| Workflow | Steps | Rough APIs | GitHub URL |
|----------|-------|------------|------------|
| Neural_Network_Training | 7 | Value, MLP, backward, parameters, zero_grad | PENDING |

---

## Workflow: Karpathy_Micrograd_Neural_Network_Training

**File:** [→](./workflows/Karpathy_Micrograd_Neural_Network_Training.md)
**Description:** End-to-end neural network training using micrograd's scalar autograd engine.
**GitHub URL:** PENDING

### Steps Overview

| # | Step Name | Rough API | Related Files |
|---|-----------|-----------|---------------|
| 1 | Data Preparation | Python lists | (user code) |
| 2 | Network Architecture Definition | `MLP.__init__`, `Layer.__init__`, `Neuron.__init__` | nn.py |
| 3 | Forward Pass Computation | `MLP.__call__`, `Layer.__call__`, `Neuron.__call__` | nn.py, engine.py |
| 4 | Loss Computation | `Value.__add__`, `Value.__mul__`, `Value.relu` | engine.py |
| 5 | Backward Pass | `Value.backward` | engine.py |
| 6 | Parameter Update | `Module.parameters`, manual update loop | nn.py |
| 7 | Training Loop Iteration | `Module.zero_grad`, repeat steps 3-6 | nn.py |

### Source Files (for enrichment)

- `micrograd/engine.py` - Core Value class with autograd operations (+, *, **, relu, backward)
- `micrograd/nn.py` - Neural network primitives (Module, Neuron, Layer, MLP)

---

### Step 1: Data_Preparation

| Attribute | Value |
|-----------|-------|
| **API Call** | Python lists/data structures (user-defined) |
| **Source Location** | (user code) - No library code |
| **External Dependencies** | None |
| **Key Parameters** | `xs: list[list[float]]` - Input feature vectors, `ys: list[float]` - Target labels |
| **Inputs** | Raw training data (e.g., from file, manual entry, or generated) |
| **Outputs** | `xs` list of input vectors, `ys` list of target values |

---

### Step 2: Network_Architecture_Definition

| Attribute | Value |
|-----------|-------|
| **API Call** | `MLP.__init__(self, nin: int, nouts: list[int]) -> None` |
| **Source Location** | `micrograd/nn.py:L47-49` |
| **External Dependencies** | `random` (stdlib), `micrograd.engine.Value` |
| **Key Parameters** | `nin: int` - Number of input features, `nouts: list[int]` - List of layer output sizes (defines architecture) |
| **Inputs** | Architecture specification: input dimension and layer sizes |
| **Outputs** | Initialized `MLP` object with random weights |

**Supporting APIs:**

| API | Signature | Source Location |
|-----|-----------|-----------------|
| `Layer.__init__` | `Layer.__init__(self, nin: int, nout: int, **kwargs) -> None` | `micrograd/nn.py:L32-33` |
| `Neuron.__init__` | `Neuron.__init__(self, nin: int, nonlin: bool = True) -> None` | `micrograd/nn.py:L15-18` |
| `Value.__init__` | `Value.__init__(self, data, _children=(), _op='') -> None` | `micrograd/engine.py:L5-11` |

---

### Step 3: Forward_Pass_Computation

| Attribute | Value |
|-----------|-------|
| **API Call** | `MLP.__call__(self, x: list[Value]) -> Value` |
| **Source Location** | `micrograd/nn.py:L51-54` |
| **External Dependencies** | `micrograd.engine.Value` |
| **Key Parameters** | `x: list[Value or float]` - Input feature vector |
| **Inputs** | Single input vector `x` from training data |
| **Outputs** | `Value` object containing prediction (scalar for single output) |

**Supporting APIs:**

| API | Signature | Source Location |
|-----|-----------|-----------------|
| `Layer.__call__` | `Layer.__call__(self, x) -> Value or list[Value]` | `micrograd/nn.py:L35-37` |
| `Neuron.__call__` | `Neuron.__call__(self, x) -> Value` | `micrograd/nn.py:L20-22` |
| `Value.__add__` | `Value.__add__(self, other) -> Value` | `micrograd/engine.py:L13-22` |
| `Value.__mul__` | `Value.__mul__(self, other) -> Value` | `micrograd/engine.py:L24-33` |
| `Value.relu` | `Value.relu(self) -> Value` | `micrograd/engine.py:L45-52` |

---

### Step 4: Loss_Computation

| Attribute | Value |
|-----------|-------|
| **API Call** | `Value.__sub__`, `Value.__pow__`, `Value.__add__` (composed for loss) |
| **Source Location** | `micrograd/engine.py:L78-79` (sub), `L35-43` (pow), `L13-22` (add) |
| **External Dependencies** | None |
| **Key Parameters** | `predictions: list[Value]` - Model outputs, `targets: list[float]` - Ground truth |
| **Inputs** | List of predictions from forward pass, list of ground truth targets |
| **Outputs** | `Value` object containing total loss (e.g., MSE or hinge loss) |

**Common Loss Patterns:**

| Pattern | Implementation | Description |
|---------|---------------|-------------|
| MSE Loss | `sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))` | Mean squared error |
| Hinge Loss | `sum((1 + -yi*scorei).relu() for yi, scorei in zip(ys, scores))` | SVM-style margin loss |

---

### Step 5: Backward_Pass

| Attribute | Value |
|-----------|-------|
| **API Call** | `Value.backward(self) -> None` |
| **Source Location** | `micrograd/engine.py:L54-70` |
| **External Dependencies** | None |
| **Key Parameters** | None (called on loss Value) |
| **Inputs** | Loss `Value` from Step 4 (contains computation graph in `_prev`) |
| **Outputs** | Gradients populated in `.grad` attribute of all `Value` objects in the graph |

**Mechanism:**
1. Builds topological ordering via DFS (L56-65)
2. Sets output gradient to 1.0 (L68)
3. Traverses in reverse, calling each node's `_backward()` closure (L69-70)

---

### Step 6: Parameter_Update

| Attribute | Value |
|-----------|-------|
| **API Call** | `Module.parameters(self) -> list[Value]` |
| **Source Location** | `micrograd/nn.py:L10-11` (base), `L56-57` (MLP override) |
| **External Dependencies** | None |
| **Key Parameters** | `learning_rate: float` - Step size for gradient descent |
| **Inputs** | Model with populated gradients from backward pass |
| **Outputs** | Updated parameter values (`p.data` modified in-place) |

**Update Pattern:**
```python
for p in model.parameters():
    p.data -= learning_rate * p.grad
```

**Supporting APIs:**

| API | Signature | Source Location |
|-----|-----------|-----------------|
| `MLP.parameters` | `MLP.parameters(self) -> list[Value]` | `micrograd/nn.py:L56-57` |
| `Layer.parameters` | `Layer.parameters(self) -> list[Value]` | `micrograd/nn.py:L39-40` |
| `Neuron.parameters` | `Neuron.parameters(self) -> list[Value]` | `micrograd/nn.py:L24-25` |

---

### Step 7: Training_Loop_Iteration

| Attribute | Value |
|-----------|-------|
| **API Call** | `Module.zero_grad(self) -> None` |
| **Source Location** | `micrograd/nn.py:L6-8` |
| **External Dependencies** | None |
| **Key Parameters** | `epochs: int` - Number of training iterations |
| **Inputs** | Model after parameter update |
| **Outputs** | Reset gradients (all `p.grad = 0`), ready for next iteration |

**Training Loop Pattern:**
```python
for epoch in range(epochs):
    # Forward pass (Step 3)
    ypred = [model(x) for x in xs]
    # Loss computation (Step 4)
    loss = compute_loss(ypred, ys)
    # Zero gradients (Step 7 - beginning of next iteration)
    model.zero_grad()
    # Backward pass (Step 5)
    loss.backward()
    # Parameter update (Step 6)
    for p in model.parameters():
        p.data -= learning_rate * p.grad
```

---

### Implementation Extraction Guide

| Step | API | Source | Dependencies | Type |
|------|-----|--------|--------------|------|
| Data_Preparation | Python lists | (user code) | None | Pattern Doc |
| Network_Architecture_Definition | `MLP.__init__` | `micrograd/nn.py:L47-49` | random, micrograd.engine | API Doc |
| Network_Architecture_Definition | `Layer.__init__` | `micrograd/nn.py:L32-33` | micrograd.engine | API Doc |
| Network_Architecture_Definition | `Neuron.__init__` | `micrograd/nn.py:L15-18` | random, micrograd.engine | API Doc |
| Network_Architecture_Definition | `Value.__init__` | `micrograd/engine.py:L5-11` | None | API Doc |
| Forward_Pass_Computation | `MLP.__call__` | `micrograd/nn.py:L51-54` | micrograd.engine | API Doc |
| Forward_Pass_Computation | `Layer.__call__` | `micrograd/nn.py:L35-37` | None | API Doc |
| Forward_Pass_Computation | `Neuron.__call__` | `micrograd/nn.py:L20-22` | None | API Doc |
| Forward_Pass_Computation | `Value.__add__` | `micrograd/engine.py:L13-22` | None | API Doc |
| Forward_Pass_Computation | `Value.__mul__` | `micrograd/engine.py:L24-33` | None | API Doc |
| Forward_Pass_Computation | `Value.relu` | `micrograd/engine.py:L45-52` | None | API Doc |
| Loss_Computation | `Value.__sub__` | `micrograd/engine.py:L78-79` | None | API Doc |
| Loss_Computation | `Value.__pow__` | `micrograd/engine.py:L35-43` | None | API Doc |
| Backward_Pass | `Value.backward` | `micrograd/engine.py:L54-70` | None | API Doc |
| Parameter_Update | `MLP.parameters` | `micrograd/nn.py:L56-57` | None | API Doc |
| Parameter_Update | `Layer.parameters` | `micrograd/nn.py:L39-40` | None | API Doc |
| Parameter_Update | `Neuron.parameters` | `micrograd/nn.py:L24-25` | None | API Doc |
| Training_Loop_Iteration | `Module.zero_grad` | `micrograd/nn.py:L6-8` | None | API Doc |

---

**Legend:** `PENDING` = GitHub repo not yet created
