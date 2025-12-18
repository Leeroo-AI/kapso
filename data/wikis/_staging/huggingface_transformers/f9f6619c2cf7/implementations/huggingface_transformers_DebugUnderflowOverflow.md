{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|huggingface_transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Debugging]], [[domain::Utilities]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
A debug utility class that detects and traces numerical instabilities (NaN/Inf) in PyTorch model training by monitoring tensor values across forward passes.

=== Description ===
The `DebugUnderflowOverflow` class helps developers identify where models start producing very large or very small values, and critically where `nan` or `inf` values appear in weights and activations. It operates by registering forward hooks on all model modules and tracking the absolute min/max values of parameters, inputs, and outputs through each layer. When overflow or underflow is detected, it provides a detailed trace of the last N frames leading up to the issue, making it easier to pinpoint the problematic layer or operation.

The tool is particularly valuable for debugging mixed-precision training (fp16/bf16) where numeric ranges are limited and overflow is more common. It helps identify layers that push values close to dtype limits before they overflow in subsequent operations.

=== Usage ===
Use this class when experiencing NaN or Inf issues during training, especially with mixed-precision (fp16/bf16). It's also useful for proactive monitoring when implementing new models or training configurations where numerical stability is uncertain. The tool supports two modes: automatic detection mode (default) and manual tracing mode for specific batches.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers huggingface_transformers]
* '''File:''' src/transformers/debug_utils.py

=== Signature ===
<syntaxhighlight lang="python">
class DebugUnderflowOverflow:
    def __init__(
        self,
        model,
        max_frames_to_save=21,
        trace_batch_nums=[],
        abort_after_batch_num=None
    ):
        """
        Args:
            model: The nn.Module to debug
            max_frames_to_save: Number of frames to keep in buffer (default: 21)
            trace_batch_nums: List of batch numbers to trace without detection
            abort_after_batch_num: Stop training after this batch number
        """

def detect_overflow(var, ctx):
    """
    Report whether a tensor contains any nan or inf entries.

    Args:
        var: The tensor variable to check
        ctx: Message to print as context

    Returns:
        True if inf or nan detected, False otherwise
    """

def get_abs_min_max(var, ctx):
    """Get formatted string of absolute min/max values for a tensor."""

class DebugOption(ExplicitEnum):
    UNDERFLOW_OVERFLOW = "underflow_overflow"
    TPU_METRICS_DEBUG = "tpu_metrics_debug"
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import DebugUnderflowOverflow
# or
from transformers.debug_utils import DebugUnderflowOverflow, detect_overflow
</syntaxhighlight>

== I/O Contract ==

=== Initialization Parameters ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| model || nn.Module || Required || The PyTorch model to monitor for numerical issues
|-
| max_frames_to_save || int || 21 || Number of forward pass frames to retain in rolling buffer
|-
| trace_batch_nums || list[int] || [] || Specific batch numbers to trace (disables automatic detection)
|-
| abort_after_batch_num || int || None || Training stops after this batch completes
|}

=== Output Format ===
{| class="wikitable"
! Output !! Type !! Description
|-
| Console Report || str || Printed trace showing abs min/max for each layer's weights, inputs, outputs
|-
| Exception || ValueError || Raised when inf/nan detected (in detection mode)
|-
| Frame Data || str || Multi-line text with module path, class name, and tensor statistics
|}

=== Detection Report Structure ===
Each frame shows:
* Fully qualified module name and class
* Absolute min/max of weights (for each named parameter)
* Absolute min/max of inputs (indexed if tuple)
* Absolute min/max of outputs (indexed if tuple)

== Usage Examples ==

=== Mode 1: Automatic Underflow/Overflow Detection ===
<syntaxhighlight lang="python">
import torch
from transformers import AutoModelForCausalLM, DebugUnderflowOverflow

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16)

# Initialize debugger (monitors all batches)
debug_overflow = DebugUnderflowOverflow(model, max_frames_to_save=50)

# Run training normally
# If nan/inf detected, will print trace and raise ValueError
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
</syntaxhighlight>

=== Mode 2: Specific Batch Tracing ===
<syntaxhighlight lang="python">
from transformers import AutoModel, DebugUnderflowOverflow

model = AutoModel.from_pretrained("bert-base-uncased")

# Trace batches 1 and 3 only (no automatic detection)
debug_overflow = DebugUnderflowOverflow(
    model,
    trace_batch_nums=[1, 3],
    abort_after_batch_num=5  # Stop after batch 5
)

# Training loop - will print full traces for batches 1 and 3
for i, batch in enumerate(dataloader):
    outputs = model(**batch)
    # Batch traces printed automatically
</syntaxhighlight>

=== Using detect_overflow Standalone ===
<syntaxhighlight lang="python">
import torch
from transformers.debug_utils import detect_overflow

# Check a tensor after some operation
tensor = torch.randn(100, 100, dtype=torch.float16)
result = tensor * 1000  # Might overflow in fp16

if detect_overflow(result, "multiplication result"):
    print("Overflow detected!")
    # Take corrective action
</syntaxhighlight>

=== Example Output ===
<syntaxhighlight lang="text">
Detected inf/nan during batch_number=0
Last 21 forward frames:
abs min  abs max  metadata
                 encoder.block.2.layer.1.DenseReluDense.wi_0 Linear
2.17e-07 4.50e+00 weight
1.79e-06 4.65e+00 input[0]
2.68e-06 3.70e+01 output
                 encoder.block.2.layer.1.DenseReluDense.wo Linear
1.01e-06 6.44e+00 weight
0.00e+00 9.74e+03 input[0]
3.18e-04 6.27e+04 output
                 encoder.block.2.layer.1.dropout Dropout
3.18e-04 6.27e+04 input[0]
0.00e+00      inf output
</syntaxhighlight>

== Implementation Details ==

=== Mechanism ===
* Uses PyTorch's `register_forward_hook` to intercept all forward passes
* Maintains a LIFO deque buffer storing the last N frames
* For each module, analyzes all parameters (weights), inputs, and outputs
* Checks tensors with `torch.isnan()` and `torch.isinf()`
* Computes absolute min/max using `.abs().min()` and `.abs().max()`

=== Performance Impact ===
The debugger slows training significantly because it:
* Registers hooks on every module
* Computes min/max statistics for every tensor
* Maintains string buffers for frame history

Therefore, disable it once debugging is complete.

=== Batch Tracking ===
* Batches are 0-indexed
* The top-level model's forward hook marks batch boundaries
* Each batch starts with a header frame showing batch number

== Related Pages ==
