# Implementation: Autocast_GradScaler

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PyTorch|https://github.com/pytorch/pytorch]]
* [[source::Doc|PyTorch Automatic Mixed Precision|https://pytorch.org/docs/stable/amp.html]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

The torch.autocast context manager and the torch.amp.GradScaler class are the two PyTorch utilities for automatic mixed-precision training. Autocast is a context manager that chooses a lower-precision dtype for operations that benefit from it, such as matrix multiplications and convolutions, while keeping numerically sensitive operations such as reductions and softmax in float32; it wraps the forward pass and the loss computation. GradScaler is a helper object that applies a scale factor to the loss before backpropagation, unscales the gradients before the optimizer step, skips any step whose gradients contain infinities or NaNs, and adjusts the factor dynamically. It is needed when autocasting to float16, whose narrow exponent range lets small gradients underflow to zero; with bfloat16, which shares float32's exponent range, the scaler is unnecessary and autocast alone is sufficient.

=== Description ===

torch.autocast and torch.amp.GradScaler are PyTorch's utilities for automatic mixed-precision training: they run selected operations in a lower-precision dtype and look after the loss during backpropagation. Together they reduce memory use and speed up training on hardware with fast 16-bit arithmetic. Which operations are kept in float32 is defined per device type; the reductions and softmax named here follow the CUDA list.

=== Usage ===

Use autocast around the forward pass in every mixed-precision training loop, and add the scaler only for float16.

== Code Reference ==

<syntaxhighlight lang="python">
torch.autocast(device_type, dtype=None, enabled=True, cache_enabled=None)
torch.amp.GradScaler(device="cuda", init_scale=65536.0, growth_factor=2.0,
                     backoff_factor=0.5, growth_interval=2000, enabled=True)
</syntaxhighlight>

== Usage Examples ==

<syntaxhighlight lang="python">
scaler = torch.amp.GradScaler("cuda")
for batch, target in loader:
    optimizer.zero_grad()
    with torch.autocast("cuda", dtype=torch.float16):
        loss = criterion(model(batch), target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
</syntaxhighlight>

== Related Pages ==

* [[requires_env::Environment:pytorch_pytorch_Single_24GB_Consumer_GPU]]
