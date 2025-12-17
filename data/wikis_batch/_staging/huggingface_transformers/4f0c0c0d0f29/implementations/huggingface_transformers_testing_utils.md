# Implementation: huggingface_transformers_testing_utils

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Contributing Tests|https://huggingface.co/docs/transformers/contributing]]
|-
! Domains
| [[domain::Testing]], [[domain::Development_Tools]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Comprehensive testing infrastructure providing decorators, fixtures, and utilities for conditionally running tests based on backend availability and hardware requirements.

=== Description ===

The `testing_utils` module (4366 lines) is the backbone of transformers' test suite. It provides:
- Backend availability decorators (`@require_torch`, `@require_tokenizers`, etc.)
- Hardware requirement decorators (`@require_torch_gpu`, `@require_accelerate`)
- Slow test markers and CI integration
- Custom doctest runner with output checking
- Test fixtures for model hubs and temporary directories
- Mocking utilities for offline testing

=== Usage ===

Use when writing or running transformers tests. Import decorators to skip tests when dependencies unavailable, use fixtures for test isolation, and leverage utilities for model comparison.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' [https://github.com/huggingface/transformers/blob/main/src/transformers/testing_utils.py src/transformers/testing_utils.py]
* '''Lines:''' 1-4366

=== Signature ===
<syntaxhighlight lang="python">
# Backend requirement decorators
def require_torch(test_case): ...
def require_torch_gpu(test_case): ...
def require_tokenizers(test_case): ...
def require_vision(test_case): ...
def require_accelerate(test_case): ...

# Slow test markers
def slow(test_case): ...
def tooslow(test_case): ...

# Test utilities
def get_tests_dir(append_path: str = None) -> str: ...
def nested_simplify(obj, decimals=3): ...
def require_torch_multi_gpu(test_case): ...

# Custom doctest
class HfDoctestModule(Module): ...
class HfDocTestParser(doctest.DocTestParser): ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers.testing_utils import (
    require_torch,
    require_torch_gpu,
    require_tokenizers,
    slow,
    get_tests_dir,
)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| test_case || Callable || Yes || Test function to decorate
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| decorated_test || Callable || Test that skips if requirements not met
|}

== Usage Examples ==

=== Skip Tests Without PyTorch ===
<syntaxhighlight lang="python">
from transformers.testing_utils import require_torch, require_torch_gpu

@require_torch
def test_model_forward():
    """Only runs if torch is available."""
    import torch
    from transformers import BertModel
    model = BertModel.from_pretrained("bert-base-uncased")
    inputs = torch.zeros(1, 10, dtype=torch.long)
    outputs = model(inputs)
    assert outputs.last_hidden_state.shape == (1, 10, 768)

@require_torch_gpu
def test_model_cuda():
    """Only runs if CUDA GPU available."""
    import torch
    from transformers import BertModel
    model = BertModel.from_pretrained("bert-base-uncased").cuda()
    inputs = torch.zeros(1, 10, dtype=torch.long, device="cuda")
    outputs = model(inputs)
</syntaxhighlight>

=== Mark Slow Tests ===
<syntaxhighlight lang="python">
from transformers.testing_utils import slow, require_torch

@slow
@require_torch
def test_full_training():
    """Skipped unless RUN_SLOW=1 environment variable set."""
    # Full training loop that takes > 1 minute
    pass
</syntaxhighlight>

== Related Pages ==
