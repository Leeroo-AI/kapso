# File: `test/test_engine.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 67 |
| Functions | `test_sanity_check`, `test_more_ops` |
| Imports | micrograd, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests that verify micrograd's forward and backward passes match PyTorch's autograd exactly.

**Mechanism:** Each test builds identical computation graphs in both micrograd and PyTorch:
- `test_sanity_check`: Tests basic operations (multiply, add, ReLU) with a single variable `x=-4.0`. Verifies both forward result and gradient match PyTorch.
- `test_more_ops`: Comprehensive test covering all operations (add, multiply, power, subtract, divide, ReLU, negation) with two variables `a=-4.0`, `b=2.0`. Uses tolerance `1e-6` for floating point comparison.

Both tests call `.backward()` on the final output and compare:
1. Forward pass: `micrograd.data == pytorch.data.item()`
2. Backward pass: `micrograd.grad == pytorch.grad.item()`

**Significance:** Critical validation that micrograd computes correct gradients. By comparing against PyTorch (the industry-standard autograd), these tests prove the implementation is mathematically correct. Essential for an educational library where correctness demonstrates the concepts.
