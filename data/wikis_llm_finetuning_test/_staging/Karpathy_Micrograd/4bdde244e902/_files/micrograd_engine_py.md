# File: `micrograd/engine.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 94 |
| Classes | `Value` |

## Understanding

**Status:** ✅ Explored

**Purpose:** Core autograd engine implementing scalar-valued automatic differentiation via the `Value` class.

**Mechanism:** The `Value` class wraps scalar numbers and tracks computational graphs. Each operation (`+`, `*`, `**`, `relu`) creates a new Value node with:
- `data`: The forward-pass result
- `grad`: Gradient storage (initialized to 0)
- `_prev`: Set of parent nodes forming the computation graph
- `_backward`: Closure computing local gradients via chain rule

The `backward()` method performs reverse-mode autodiff:
1. Builds topological ordering via DFS from output to inputs
2. Sets output gradient to 1.0
3. Traverses in reverse order, calling each node's `_backward()` to propagate gradients

Supports operations: add, multiply, power, ReLU, negate, subtract, divide (via `x * y**-1`).

**Significance:** This is the heart of micrograd - a minimal but complete autograd implementation. It demonstrates how PyTorch-style automatic differentiation works at a fundamental level. All neural network functionality in `nn.py` builds on this class.
