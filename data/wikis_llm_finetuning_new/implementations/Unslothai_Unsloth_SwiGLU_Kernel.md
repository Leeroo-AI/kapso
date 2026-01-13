# Implementation: SwiGLU Kernel

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Kernels]], [[domain::GPU_Optimization]], [[domain::Activation_Functions]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

The SwiGLU kernel provides a high-performance Triton implementation of the SwiGLU (Swish-Gated Linear Unit) activation function used in modern transformer architectures like LLaMA, PaLM, and GPT-NeoX. SwiGLU combines the Swish activation (SiLU) with a gating mechanism for improved model performance.

The SwiGLU formula is:
<code>SwiGLU(e, g) = (e * sigmoid(e)) * g = SiLU(e) * g</code>

The implementation includes:
* '''Forward kernel''' (<code>_fg_kernel</code>) - Computes the SwiGLU activation
* '''Backward/Derivative kernel''' (<code>_DWf_DW_dfg_kernel</code>) - Computes gradients for backpropagation
* '''Long indexing support''' - Handles tensors exceeding int32 element limits

== Code Reference ==

'''File:''' <code>unsloth/kernels/swiglu.py</code>

=== Constants for Safe Indexing ===

<syntaxhighlight lang="python">
# signed int32 max is 2**31-1 so num_elements cannot exceed 2**31
NUM_INT32_ELEMENTS = 2**31
SAFE_INT32_BUFFER_MULTIPLIER = 4
BLOCK_SIZE = 1024
INT32_SAFETY_BUFFER = NUM_INT32_ELEMENTS - BLOCK_SIZE * SAFE_INT32_BUFFER_MULTIPLIER
</syntaxhighlight>

=== Forward Kernel ===

<syntaxhighlight lang="python">
@triton.jit
def _fg_kernel(
    e,
    g,
    h,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
):
    block_idx = tl.program_id(0)
    if LONG_INDEXING:
        offsets = block_idx.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        n_elements = tl.cast(n_elements, tl.int64)
    else:
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    # f = e * sigmoid(e) = SiLU(e)
    f_row = e_row * tl.sigmoid(e_row)
    f_row = f_row.to(g_row.dtype)
    # h = f * g = SwiGLU output
    h_row = f_row * g_row

    tl.store(h + offsets, h_row, mask=mask)
</syntaxhighlight>

=== Backward/Derivative Kernel ===

<syntaxhighlight lang="python">
@triton.jit
def _DWf_DW_dfg_kernel(
    DW,
    e,
    g,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
):
    """
    Computes forward values and derivatives in a fused manner:
    e = e.float()
    se = 1.0 / (1.0 + torch.exp(-e))  # sigmoid
    f = (se * e).to(dtype)            # SiLU
    h = f * g                          # SwiGLU output
    df = DW * f                        # derivative w.r.t. f
    dg = DW * g                        # derivative w.r.t. g
    de = (dg.float() * se * (1.0 + e * (1.0 - se))).to(dtype)  # derivative w.r.t. e
    """
    block_idx = tl.program_id(0)
    ...

    # Compute derivatives
    se_row = tl.sigmoid(e_row)
    f_row = se_row * e_row
    f_row = f_row.to(DW_row.dtype)
    h_row = f_row * g_row
    df_row = DW_row * f_row
    dg_row = DW_row * g_row
    de_row = dg_row.to(tl.float32) * se_row * (1.0 + e_row * (1.0 - se_row))
    de_row = de_row.to(DW_row.dtype)

    # Store derivatives in-place (reuses input buffers)
    tl.store(DW + offsets, h_row, mask=mask)   # h = f * g
    tl.store(e + offsets, df_row, mask=mask)   # df = DW * f
    tl.store(g + offsets, de_row, mask=mask)   # de
</syntaxhighlight>

=== Python Wrapper Functions ===

<syntaxhighlight lang="python">
def swiglu_fg_kernel(e, g):
    batch, seq_len, hd = e.shape
    n_elements = e.numel()
    h = torch.empty((batch, seq_len, hd), dtype=e.dtype, device=e.device)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_gpu_device(e.device):
        _fg_kernel[grid](
            e, g, h, n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            LONG_INDEXING=0 if n_elements <= INT32_SAFETY_BUFFER else 1,
        )
    return h


def swiglu_DWf_DW_dfg_kernel(DW, e, g):
    batch_seq_len, hd = e.shape
    n_elements = e.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_gpu_device(e.device):
        _DWf_DW_dfg_kernel[grid](
            DW, e, g, n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            LONG_INDEXING=0 if n_elements <= INT32_SAFETY_BUFFER else 1,
        )
    return DW, e, g  # Modified in-place
</syntaxhighlight>

== I/O Contract ==

=== swiglu_fg_kernel (Forward) ===

'''Signature:'''
<syntaxhighlight lang="python">
def swiglu_fg_kernel(e: torch.Tensor, g: torch.Tensor) -> torch.Tensor
</syntaxhighlight>

'''Inputs:'''
{| class="wikitable"
|-
! Parameter !! Type !! Description
|-
| <code>e</code> || <code>torch.Tensor</code> || First input (gate), shape <code>(batch, seq_len, hidden_dim)</code>
|-
| <code>g</code> || <code>torch.Tensor</code> || Second input (value), shape <code>(batch, seq_len, hidden_dim)</code>
|}

'''Outputs:'''
{| class="wikitable"
|-
! Return !! Type !! Description
|-
| <code>h</code> || <code>torch.Tensor</code> || SwiGLU output: <code>SiLU(e) * g</code>, same shape as inputs
|}

'''Constraints:'''
* <code>e</code> and <code>g</code> must have the same shape
* Automatically switches to int64 indexing for tensors with > ~2B elements

=== swiglu_DWf_DW_dfg_kernel (Backward) ===

'''Signature:'''
<syntaxhighlight lang="python">
def swiglu_DWf_DW_dfg_kernel(
    DW: torch.Tensor,
    e: torch.Tensor,
    g: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
</syntaxhighlight>

'''Inputs:'''
{| class="wikitable"
|-
! Parameter !! Type !! Description
|-
| <code>DW</code> || <code>torch.Tensor</code> || Upstream gradient, shape <code>(batch*seq_len, hidden_dim)</code>
|-
| <code>e</code> || <code>torch.Tensor</code> || First input (gate), shape <code>(batch*seq_len, hidden_dim)</code>
|-
| <code>g</code> || <code>torch.Tensor</code> || Second input (value), shape <code>(batch*seq_len, hidden_dim)</code>
|}

'''Outputs (modified in-place):'''
{| class="wikitable"
|-
! Return !! Type !! Description
|-
| <code>DW</code> || <code>torch.Tensor</code> || Forward output <code>h = f * g</code> (stored in DW buffer)
|-
| <code>e</code> || <code>torch.Tensor</code> || Gradient <code>df = DW * f</code> (stored in e buffer)
|-
| <code>g</code> || <code>torch.Tensor</code> || Gradient <code>de</code> (stored in g buffer)
|}

'''Note:''' This kernel is designed for memory efficiency by reusing input buffers for output storage.

== Usage Examples ==

=== Basic SwiGLU Forward Pass ===

<syntaxhighlight lang="python">
from unsloth.kernels.swiglu import swiglu_fg_kernel

# Typical MLP intermediate dimensions
batch, seq_len, hidden_dim = 2, 1024, 11008

# Two projections from attention output
e = torch.randn(batch, seq_len, hidden_dim, dtype=torch.float16, device="cuda")
g = torch.randn(batch, seq_len, hidden_dim, dtype=torch.float16, device="cuda")

# Compute SwiGLU: SiLU(e) * g
h = swiglu_fg_kernel(e, g)
print(h.shape)  # (2, 1024, 11008)
</syntaxhighlight>

=== Integration with MLP Layer ===

<syntaxhighlight lang="python">
class FastMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        # Compute gate and up projections
        gate = self.gate_proj(x)  # e in SwiGLU
        up = self.up_proj(x)      # g in SwiGLU

        # Apply fast SwiGLU kernel
        hidden = swiglu_fg_kernel(gate, up)

        # Project back to hidden size
        return self.down_proj(hidden)
</syntaxhighlight>

=== Backward Pass with Gradient Computation ===

<syntaxhighlight lang="python">
from unsloth.kernels.swiglu import swiglu_DWf_DW_dfg_kernel

batch_seq_len, hidden_dim = 2048, 11008

# Upstream gradient and saved tensors
DW = torch.randn(batch_seq_len, hidden_dim, dtype=torch.float16, device="cuda")
e = torch.randn(batch_seq_len, hidden_dim, dtype=torch.float16, device="cuda")
g = torch.randn(batch_seq_len, hidden_dim, dtype=torch.float16, device="cuda")

# Compute derivatives (in-place modification)
h_out, df, de = swiglu_DWf_DW_dfg_kernel(DW.clone(), e.clone(), g.clone())

# h_out = SwiGLU forward output (for verification)
# df = gradient w.r.t. f (intermediate)
# de = gradient w.r.t. e (input)
</syntaxhighlight>

=== Large Tensor Handling ===

<syntaxhighlight lang="python">
# For very large tensors (>2B elements), long indexing is automatic
large_batch = 64
large_seq = 8192
intermediate = 14336  # ~7.5B elements total

e = torch.randn(large_batch, large_seq, intermediate, dtype=torch.bfloat16, device="cuda")
g = torch.randn(large_batch, large_seq, intermediate, dtype=torch.bfloat16, device="cuda")

# Kernel automatically uses int64 indexing when needed
h = swiglu_fg_kernel(e, g)  # LONG_INDEXING=1 internally
</syntaxhighlight>

== Related Pages ==

* [[Unslothai_Unsloth_Kernel_Utils]] - Utility functions including <code>calculate_settings</code> and <code>torch_gpu_device</code>
* [[Unslothai_Unsloth_RMSNorm_Kernel]] - RMS layer normalization kernel
* [[Unslothai_Unsloth_RoPE_Kernel]] - Rotary position embedding kernel
