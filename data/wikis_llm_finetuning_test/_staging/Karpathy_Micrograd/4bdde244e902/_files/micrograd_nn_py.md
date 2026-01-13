# File: `micrograd/nn.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 60 |
| Classes | `Module`, `Neuron`, `Layer`, `MLP` |
| Imports | micrograd, random |

## Understanding

**Status:** ✅ Explored

**Purpose:** Neural network library providing PyTorch-like building blocks (`Module`, `Neuron`, `Layer`, `MLP`) on top of the autograd engine.

**Mechanism:**
- `Module`: Base class with `zero_grad()` to reset all parameter gradients and `parameters()` to collect trainable weights
- `Neuron`: Single neuron with `nin` input weights (randomly initialized in [-1,1]), a bias term, and optional ReLU activation. Forward pass: `sum(w*x) + b`, optionally followed by ReLU
- `Layer`: Collection of `nout` neurons, each with `nin` inputs. Forward pass applies all neurons to input, returns list (or single value if nout=1)
- `MLP`: Multi-layer perceptron built from consecutive Layers. Constructor takes `nin` (input size) and `nouts` list (output sizes per layer). Final layer is linear (no ReLU); hidden layers use ReLU

**Significance:** Demonstrates how a PyTorch-like neural network API is built on top of autograd. The simple hierarchy (Module → Neuron → Layer → MLP) mirrors PyTorch's `nn.Module` pattern. Used for educational examples like training a classifier.
