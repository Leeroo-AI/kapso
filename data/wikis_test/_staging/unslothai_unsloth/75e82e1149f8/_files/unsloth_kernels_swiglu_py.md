# File: `unsloth/kernels/swiglu.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 143 |
| Functions | `swiglu_fg_kernel`, `swiglu_DWf_DW_dfg_kernel` |
| Imports | torch, triton, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Optimized SwiGLU (Swish-Gated Linear Unit) activation function for transformer MLP layers.

**Mechanism:** Implements the SwiGLU activation: f = e * sigmoid(e), output = f * g, where e is gate projection and g is up projection. The forward kernel (`swiglu_fg_kernel`) computes f using sigmoid and multiplies with g. The backward kernel (`swiglu_DWf_DW_dfg_kernel`) computes gradients: df = DW * f, dg = DW * g, de = dg * sigmoid(e) * (1 + e * (1 - sigmoid(e))). The derivative formula leverages the property that d(x*sigmoid(x))/dx = sigmoid(x) * (1 + x*(1-sigmoid(x))). Handles large tensors via conditional int64 indexing when element count exceeds 2³¹.

**Significance:** SwiGLU is the activation function used in LLaMA and many state-of-the-art models, providing better performance than ReLU or GELU. This fused implementation combines the gate activation and element-wise multiplication in single kernels, reducing memory bandwidth requirements substantially. The careful derivative computation enables efficient backpropagation. The large tensor support ensures compatibility with models of any size. This kernel is used extensively in LoRA training via the fast_lora module.
